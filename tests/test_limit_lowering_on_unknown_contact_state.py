"""Tests for the `limit_lowering_on_unknown_contact_state` toggle
(`auto_ventilate_options`).

Default: an unreadable tilted contact reads as closed; an unreadable opened
contact reads as not open - both assumptions, not confirmations. With the toggle
on, an unreadable TILTED reading resolves as 'tlt' (the safe VENT floor either
way), consumed at 5 sites: effective_state.w, recovered_window,
window_tilted_now, both contact_missing gates.

The OPENED contact gets a narrower extension - window_opened_now_or_unknown,
consumed only by lockout_now.* - withholding a close/shading-start/end without
ever driving. Every other opened-contact consumer (contact handler, LOCKOUT's
s_opn, resident chains, recovery_target) and the readiness gates stay
confirmed-only/unrelaxed (see design-decisions.md).

Cascade-level coverage lives in test_restart_recovery.py
(TestContactGate/TestCascadeParity/TestRecoveredWindow/TestResumeTrigger).
This file covers window_tilted_now/window_opened_now_or_unknown and their
live-branch consumers (lockout_now and friends), none of which go through
effective_state.

Run with: pytest tests/ -v
"""
import pathlib
import types

import jinja2
import pytest
import yaml


BLUEPRINT_PATH = (
    pathlib.Path(__file__).parent.parent
    / "blueprints"
    / "automation"
    / "cover_control_automation.yaml"
)

INVALID_STATES = ["", "unavailable", "unknown", "none", "None", "null", "query failed", []]


def _load_blueprint_yaml() -> dict:
    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor("!input", lambda loader, node: loader.construct_scalar(node))
    with open(BLUEPRINT_PATH, encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader)  # noqa: S506


BP = _load_blueprint_yaml()


def _find_branch_by_alias(node, alias: str):
    if isinstance(node, dict):
        if node.get("alias") == alias:
            return node
        for value in node.values():
            found = _find_branch_by_alias(value, alias)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_branch_by_alias(item, alias)
            if found is not None:
                return found
    return None


def _find_variable_definition(node, name: str):
    """Deep search for a `variables:` entry anywhere in the action tree - unlike
    _action_var, this also reaches into nested choose/sequence branches (needed for
    recovery_target, which lives inside the force-disable branch, not at the top
    action-level variables: steps)."""
    if isinstance(node, dict):
        if name in node and not isinstance(node[name], (dict, list)):
            return node[name]
        for value in node.values():
            found = _find_variable_definition(value, name)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_variable_definition(item, name)
            if found is not None:
                return found
    return None


def _action_var(name: str):
    """Pull one variable out of the action-level `variables:` steps - the same
    "EVENT NORMALIZATION" block window_opened_now/window_tilted_now/lockout_now
    live in."""
    for step in BP["actions"]:
        if isinstance(step, dict) and name in step.get("variables", {}):
            return step["variables"][name]
    raise AssertionError(f"action-level variable {name!r} not found")


def _render(template, entity_states: dict, group_members: dict | None = None, **variables) -> str:
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    env.globals["states"] = lambda entity_id: entity_states.get(entity_id, "unknown")
    # expand(): a plain entity passes through as its own single-item state list; a
    # registered "group" (group_members) flattens to its members' states instead -
    # mirrors real HA (expand() recurses into anything exposing entity_id membership,
    # passes everything else through unchanged).
    members = group_members or {}

    def expand(entity_id):
        if isinstance(entity_id, list):
            return []
        group = members.get(entity_id)
        if group is not None:
            return [types.SimpleNamespace(state=entity_states.get(m, "unknown")) for m in group]
        return [types.SimpleNamespace(state=entity_states.get(entity_id, "unknown"))]

    env.globals["expand"] = expand
    return env.from_string(template).render(**variables).strip()


def _render_bool(template, entity_states: dict, group_members: dict | None = None, **variables) -> bool:
    out = _render(template, entity_states, group_members, **variables)
    if out == "True":
        return True
    if out == "False":
        return False
    return bool(out)


# ─────────────────────────────────────────────────────────────────────────────
# The input option and its early trigger_variables flag
# ─────────────────────────────────────────────────────────────────────────────


class TestInputOption:
    def test_option_offered_in_ventilation_configuration(self):
        """Lives in auto_ventilate_options, not the general individual_config/prevent_*
        bucket: every sibling option there is ventilation-specific and gated the same
        way (is_ventilation_enabled) - see design-decisions.md."""
        selector = (
            BP["blueprint"]["input"]["contacts_section"]["input"]
            ["auto_ventilate_options"]["selector"]["select"]
        )
        assert "limit_lowering_on_unknown_contact_state" in [o["value"] for o in selector["options"]]

    def test_flag_lives_in_trigger_variables(self):
        """effective_state/recovered_window read it, so - like shading_over_ventilation,
        its closest sibling - it must be resolved before the action scope, and it is
        pure list membership, which the limited trigger_variables context allows
        (Invariant 10)."""
        assert "limit_lowering_on_unknown_contact_state" in BP["trigger_variables"]
        assert "states(" not in BP["trigger_variables"]["limit_lowering_on_unknown_contact_state"]
        assert "limit_lowering_on_unknown_contact_state" not in BP["variables"]

    def test_default_is_off(self):
        """No option configured = the existing behaviour, unchanged."""
        template = BP["trigger_variables"]["limit_lowering_on_unknown_contact_state"]
        assert _render(template, {}, auto_ventilate_options=[]) == "False"

    def test_on_when_selected(self):
        template = BP["trigger_variables"]["limit_lowering_on_unknown_contact_state"]
        assert _render(template, {}, auto_ventilate_options=["limit_lowering_on_unknown_contact_state"]) == "True"


# ─────────────────────────────────────────────────────────────────────────────
# window_tilted_now: the second, independent read of the tilt contact that live
# Opening/Closing/Shading/Resident/Force-disable branches dispatch on directly -
# NOT through effective_state (see architecture.md's event-normalization block).
# ─────────────────────────────────────────────────────────────────────────────


class TestWindowTiltedNow:
    TPL = staticmethod(lambda: _action_var("window_tilted_now"))

    def _run(self, tilted_state, *, unknown_tilt_ok=False, configured=True,
             tilted_entities=None, group_members=None):
        entities = {"binary_sensor.tilted": tilted_state}
        entities.update(tilted_entities or {})
        shared = dict(
            contact_window_tilted="binary_sensor.tilted" if configured else [],
            invalid_states=INVALID_STATES,
            limit_lowering_on_unknown_contact_state=unknown_tilt_ok,
        )
        # window_tilted_now references window_tilted_confirmed and the shared
        # top-level tilted_invalid by name - render them first, like the blueprint's
        # own evaluation order.
        tilted_invalid = _render_bool(
            BP["variables"]["tilted_invalid"], entities, group_members, **shared
        )
        window_tilted_confirmed = _render_bool(
            _action_var("window_tilted_confirmed"), entities, group_members, **shared
        )
        return _render_bool(
            self.TPL(), entities, group_members,
            tilted_invalid=tilted_invalid,
            window_tilted_confirmed=window_tilted_confirmed,
            **shared,
        )

    @pytest.mark.parametrize("state", ["on", "true"])
    def test_real_tilted_reading_is_unaffected(self, state):
        assert self._run(state, unknown_tilt_ok=False) is True
        assert self._run(state, unknown_tilt_ok=True) is True

    @pytest.mark.parametrize("state", ["off", "false"])
    def test_real_closed_reading_is_unaffected(self, state):
        assert self._run(state, unknown_tilt_ok=False) is False
        assert self._run(state, unknown_tilt_ok=True) is False

    @pytest.mark.parametrize("state", ["unknown", "unavailable", "", "none"])
    def test_toggle_off_is_byte_for_byte_unchanged(self, state):
        assert self._run(state, unknown_tilt_ok=False) is False

    @pytest.mark.parametrize("state", ["unknown", "unavailable", "", "none"])
    def test_toggle_on_resolves_as_tilted(self, state):
        assert self._run(state, unknown_tilt_ok=True) is True

    def test_unconfigured_contact_is_unaffected(self):
        assert self._run("unavailable", unknown_tilt_ok=True, configured=False) is False

    # contact_window_tilted may be a "binary sensor group" aggregating several
    # physical sensors (domain binary_sensor, so it passes the entity selector). A
    # single dropped member does not make the group itself unavailable - HA only
    # reports that once EVERY member is - so the group's own state can stay a
    # determinate on/off while masking one window. Unlike the readiness gate (always
    # checked), this detection is toggle-gated: without the toggle, window_tilted_now
    # had no invalid-state handling of its own before this feature existed at all.
    def test_group_member_dropout_detected_when_toggle_is_on(self):
        assert self._run(
            "off", unknown_tilt_ok=True,
            tilted_entities={"binary_sensor.tilted_1": "off", "binary_sensor.tilted_2": "unavailable"},
            group_members={"binary_sensor.tilted": ["binary_sensor.tilted_1", "binary_sensor.tilted_2"]},
        ) is True

    def test_group_member_dropout_ignored_when_toggle_is_off(self):
        assert self._run(
            "off", unknown_tilt_ok=False,
            tilted_entities={"binary_sensor.tilted_1": "off", "binary_sensor.tilted_2": "unavailable"},
            group_members={"binary_sensor.tilted": ["binary_sensor.tilted_1", "binary_sensor.tilted_2"]},
        ) is False

    def test_healthy_group_is_unaffected(self):
        assert self._run(
            "off", unknown_tilt_ok=True,
            tilted_entities={"binary_sensor.tilted_1": "off", "binary_sensor.tilted_2": "off"},
            group_members={"binary_sensor.tilted": ["binary_sensor.tilted_1", "binary_sensor.tilted_2"]},
        ) is False


# ─────────────────────────────────────────────────────────────────────────────
# The live branches that read window_tilted_now (and thus never go through
# effective_state) must actually be wired to it - a fix to the flag alone is
# only real if these still consume it.
# ─────────────────────────────────────────────────────────────────────────────


class TestBranchesConsumeTheFlag:
    @pytest.mark.parametrize("alias,needle", [
        ("Window tilted. No lockout. Move to ventilation position instead of closing",
         "{{ window_tilted_now }}"),
        ("Shading start - hold ventilation floor (window tilted)", "{{ window_tilted_now }}"),
        ("Ventilation after shading ends", "{{ window_tilted_now }}"),
        ("Resident leaving: target VENTILATION (window tilted)", "{{ window_tilted_now }}"),
        ("Resident arriving: window tilted → hold ventilation position",
         "{{ window_tilted_now }}"),
    ])
    def test_branch_condition_reads_the_flag(self, alias, needle):
        branch = _find_branch_by_alias(BP["actions"], alias)
        assert branch is not None, f"branch {alias!r} not found"
        assert needle in branch["conditions"]

    def test_lockout_now_derives_from_the_confirmed_only_flag(self):
        """lockout_now must NOT read window_tilted_now (which also counts an
        unknown reading as tilted when the toggle is on) - LOCKOUT escalating on
        an unconfirmed reading is exactly the wrong-assumption-direction outcome this
        feature exists to avoid. It reads window_tilted_confirmed (a real reading
        only) instead."""
        lockout_now = _action_var("lockout_now")
        for key in ("closing", "shading_start", "shading_end"):
            assert "window_tilted_confirmed" in lockout_now[key]
            assert "window_tilted_now" not in lockout_now[key]

    def test_window_tilted_confirmed_ignores_the_toggle(self):
        """Regression guard: window_tilted_confirmed must stay the plain real-reading
        check, byte-for-byte the pre-feature window_tilted_now definition, so it can
        never be widened by limit_lowering_on_unknown_contact_state the way window_tilted_now is."""
        template = _action_var("window_tilted_confirmed")
        assert "limit_lowering_on_unknown_contact_state" not in template
        assert "invalid_states" not in template
        assert "expand(" not in template

    def test_closing_lockout_leaf_reads_lockout_now(self):
        branch = _find_branch_by_alias(BP["actions"], "Lockout protection when closing")
        assert "{{ lockout_now.closing }}" in branch["conditions"]

    def test_force_disable_recovery_target_reads_the_flag(self):
        recovery_target = _find_variable_definition(BP["actions"], "recovery_target")
        assert recovery_target is not None
        assert "window_tilted_now" in recovery_target


# ─────────────────────────────────────────────────────────────────────────────
# Regression: an unknown tilt reading must never escalate to LOCKOUT via the
# pre-existing lockout_tilted_when_* options - LOCKOUT's assumed direction
# (fully open) is not safe under both truths. Also covers the opened contact's
# own narrower withhold-only extension (window_opened_now or (toggle and opened_invalid)).
# ─────────────────────────────────────────────────────────────────────────────


class TestLockoutDoesNotEscalateOnUnknownTilt:
    def _lockout_now(self, tilted_state, *, unknown_tilt_ok, lockout_tilted_when_closing,
                      opened_state="off", opened_entities=None, group_members=None):
        entities = {"binary_sensor.tilted": tilted_state, "binary_sensor.opened": opened_state}
        entities.update(opened_entities or {})
        shared = dict(
            contact_window_opened="binary_sensor.opened",
            contact_window_tilted="binary_sensor.tilted",
            invalid_states=INVALID_STATES,
            limit_lowering_on_unknown_contact_state=unknown_tilt_ok,
        )
        # lockout_now.closing references window_opened_now/opened_invalid/
        # window_tilted_confirmed by name (computed once, earlier in the real
        # blueprint) - render them first, like the blueprint does.
        window_opened_now = _render_bool(
            _action_var("window_opened_now"), entities, group_members, **shared
        )
        opened_invalid = _render_bool(
            BP["variables"]["opened_invalid"], entities, group_members, **shared
        )
        window_tilted_confirmed = _render_bool(
            _action_var("window_tilted_confirmed"), entities, group_members, **shared
        )
        template = _action_var("lockout_now")["closing"]
        return _render_bool(
            template, entities,
            window_opened_now=window_opened_now,
            opened_invalid=opened_invalid,
            window_tilted_confirmed=window_tilted_confirmed,
            lockout_tilted_when_closing=lockout_tilted_when_closing,
            **shared,
        )

    def test_unknown_tilt_does_not_trigger_lockout_even_with_the_option_on(self):
        assert self._lockout_now(
            "unavailable", unknown_tilt_ok=True, lockout_tilted_when_closing=True,
        ) is False

    def test_a_real_tilted_reading_still_triggers_lockout_as_before(self):
        """The pre-existing lockout_tilted_when_closing option must keep working
        for a genuine, confirmed tilted reading - only the unknown-fallback is
        excluded."""
        assert self._lockout_now(
            "on", unknown_tilt_ok=True, lockout_tilted_when_closing=True,
        ) is True

    def test_window_tilted_now_itself_is_unaffected_by_the_fix(self):
        """The VENT floor must still treat the unknown reading as tilted - only
        the LOCKOUT escalation path was narrowed."""
        window_tilted_now = _action_var("window_tilted_now")
        entities = {"binary_sensor.tilted": "unavailable"}
        shared = dict(
            contact_window_tilted="binary_sensor.tilted",
            invalid_states=INVALID_STATES,
            limit_lowering_on_unknown_contact_state=True,
        )
        assert _render_bool(
            window_tilted_now, entities,
            tilted_invalid=_render_bool(BP["variables"]["tilted_invalid"], entities, **shared),
            window_tilted_confirmed=_render_bool(
                _action_var("window_tilted_confirmed"), entities, **shared),
            **shared,
        ) is True

    # ─────────────────────────────────────────────────────────────────────
    # The narrower opened-contact extension: withhold-only via lockout_now,
    # never a proactive drive. window_opened_now itself stays confirmed-only.
    # ─────────────────────────────────────────────────────────────────────

    def test_unknown_opened_withholds_closing_even_without_any_lockout_tilted_option(self):
        """The opened contact's lockout is unconditional (matches its own
        pre-existing behaviour for a confirmed 'on' reading) - no
        lockout_tilted_when_* option needs to be enabled."""
        assert self._lockout_now(
            "off", unknown_tilt_ok=True, lockout_tilted_when_closing=False,
            opened_state="unavailable",
        ) is True

    def test_unknown_opened_is_ignored_when_toggle_is_off(self):
        assert self._lockout_now(
            "off", unknown_tilt_ok=False, lockout_tilted_when_closing=False,
            opened_state="unavailable",
        ) is False

    def test_group_member_dropout_on_the_opened_contact_also_withholds(self):
        assert self._lockout_now(
            "off", unknown_tilt_ok=True, lockout_tilted_when_closing=False,
            opened_state="off",
            opened_entities={"binary_sensor.opened_1": "off", "binary_sensor.opened_2": "unavailable"},
            group_members={"binary_sensor.opened": ["binary_sensor.opened_1", "binary_sensor.opened_2"]},
        ) is True

    def test_window_opened_now_itself_is_unaffected_by_the_fix(self):
        """Every OTHER consumer of window_opened_now (the contact handler's own
        drive, effective_state's LOCKOUT layer, the resident target chains,
        force-disable recovery_target) must keep requiring a real reading -
        only lockout_now gets the toggle-aware variable."""
        window_opened_now = _action_var("window_opened_now")
        assert "limit_lowering_on_unknown_contact_state" not in window_opened_now
        assert "invalid_states" not in window_opened_now
        assert "expand(" not in window_opened_now

    def test_lockout_now_reads_the_toggle_aware_opened_variable(self):
        lockout_now = _action_var("lockout_now")
        for key in ("closing", "shading_start", "shading_end"):
            assert "opened_invalid" in lockout_now[key]
            assert "limit_lowering_on_unknown_contact_state" in lockout_now[key]


# ─────────────────────────────────────────────────────────────────────────────
# Bug fix: lockout_now.* can fire from an unconfirmed opened-contact reading, but
# win must only assert "opn" when actually confirmed - otherwise a merely unknown
# reading gets remembered as confirmed-open, and the (deliberately unrelaxed)
# opened-contact readiness gate then freezes on that false memory.
# ─────────────────────────────────────────────────────────────────────────────


BRANCH_FLAG = [
    ("Lockout protection when closing", "closing", "lockout_tilted_when_closing"),
    ("Consider lockout protection when shading starts", "shading_start", "lockout_tilted_when_shading_starts"),
    ("Lockout protection when shading ends", "shading_end", "lockout_tilted_when_shading_ends"),
]


def _win_template(alias):
    branch = _find_branch_by_alias(BP["actions"], alias)
    assert branch is not None, f"branch {alias!r} not found"
    return branch["sequence"][0]["variables"]["update_values"]["win"]


def _win_on_lockout_template(key):
    """The 3 win: ternaries were factored into one shared win_on_lockout dict
    (same scope as lockout_now itself, see design-decisions.md) - this is now the
    single source of truth for the ternary; _win_template(alias) just references it."""
    win_on_lockout = _action_var("win_on_lockout")
    assert key in win_on_lockout, f"win_on_lockout.{key} not found"
    return win_on_lockout[key]


class TestWinFieldOnLockoutBranches:
    @pytest.mark.parametrize("alias,key,flag", BRANCH_FLAG)
    def test_branch_win_field_references_win_on_lockout(self, alias, key, flag):
        assert _win_template(alias) == "{{ win_on_lockout." + key + " }}"

    @pytest.mark.parametrize("alias,key,flag", BRANCH_FLAG)
    def test_confirmed_open_still_asserts_opn(self, alias, key, flag):
        assert _render(
            _win_on_lockout_template(key), {}, window_opened_now=True,
            window_tilted_confirmed=False, helper_state_window="cls",
            **{flag: False},
        ) == "opn"

    @pytest.mark.parametrize("alias,key,flag", BRANCH_FLAG)
    def test_confirmed_tilted_lockout_extension_still_asserts_opn(self, alias, key, flag):
        """Pre-existing, unrelated case (the lockout_tilted_when_* options extend
        LOCKOUT to a confirmed tilted reading) - not part of the bug, must not change."""
        assert _render(
            _win_on_lockout_template(key), {}, window_opened_now=False,
            window_tilted_confirmed=True, helper_state_window="cls",
            **{flag: True},
        ) == "opn"

    @pytest.mark.parametrize("alias,key,flag", BRANCH_FLAG)
    @pytest.mark.parametrize("preserved", ["cls", "tlt"])
    def test_unconfirmed_fallback_preserves_helper_state_window(self, alias, key, flag, preserved):
        """The bug: when lockout_now.<key> fires purely via the toggle's
        unconfirmed-opened fallback, win must preserve the persisted value,
        never assert 'opn' for a reading that was never confirmed."""
        assert _render(
            _win_on_lockout_template(key), {}, window_opened_now=False,
            window_tilted_confirmed=False, helper_state_window=preserved,
            **{flag: False},
        ) == preserved


class TestBug1WinNotAssertedOnUnconfirmedFallback:
    """End-to-end: chains the real action-scope variables in the blueprint's own
    evaluation order (window_opened_now -> window_opened_now_or_unknown ->
    lockout_now.<key> -> the win: template), mirroring
    TestLockoutDoesNotEscalateOnUnknownTilt._lockout_now above."""

    def _win_for(self, alias, key, flag, *, opened_state, tilted_state="off",
                 lockout_tilted, helper_state_window, unknown_ok=True):
        entities = {"binary_sensor.opened": opened_state, "binary_sensor.tilted": tilted_state}
        shared = dict(
            contact_window_opened="binary_sensor.opened",
            contact_window_tilted="binary_sensor.tilted",
            invalid_states=INVALID_STATES,
            limit_lowering_on_unknown_contact_state=unknown_ok,
        )
        window_opened_now = _render_bool(_action_var("window_opened_now"), entities, **shared)
        opened_invalid = _render_bool(BP["variables"]["opened_invalid"], entities, **shared)
        window_tilted_confirmed = _render_bool(_action_var("window_tilted_confirmed"), entities, **shared)
        lockout_fired = _render_bool(
            _action_var("lockout_now")[key], entities,
            window_opened_now=window_opened_now,
            opened_invalid=opened_invalid,
            window_tilted_confirmed=window_tilted_confirmed,
            **{flag: lockout_tilted},
            **shared,
        )
        win_on_lockout_value = _render(
            _win_on_lockout_template(key), entities,
            window_opened_now=window_opened_now,
            window_tilted_confirmed=window_tilted_confirmed,
            helper_state_window=helper_state_window,
            **{flag: lockout_tilted},
        )
        # win_on_lockout is a single dict computed once for all 3 keys - the branch's
        # own win: field just references win_on_lockout.<key> by name.
        win = _render(
            _win_template(alias), entities,
            win_on_lockout={key: win_on_lockout_value},
        )
        return lockout_fired, win

    @pytest.mark.parametrize("alias,key,flag", BRANCH_FLAG)
    def test_unconfirmed_opened_fallback_fires_lockout_but_preserves_win(self, alias, key, flag):
        lockout_fired, win = self._win_for(
            alias, key, flag, opened_state="unavailable",
            lockout_tilted=False, helper_state_window="cls",
        )
        assert lockout_fired is True   # sanity: the branch really does fire
        assert win == "cls"            # ...but win is preserved, not asserted 'opn'

    @pytest.mark.parametrize("alias,key,flag", BRANCH_FLAG)
    def test_confirmed_open_still_fires_lockout_and_asserts_opn(self, alias, key, flag):
        lockout_fired, win = self._win_for(
            alias, key, flag, opened_state="on",
            lockout_tilted=False, helper_state_window="cls",
        )
        assert lockout_fired is True
        assert win == "opn"


# ─────────────────────────────────────────────────────────────────────────────
# Config validator: the toggle is inert without Ventilation Mode / a configured
# tilted contact, and should say so.
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigValidator:
    def _checks(self):
        return BLUEPRINT_PATH.read_text(encoding="utf-8")

    def test_warns_without_ventilation_mode(self):
        text = self._checks()
        assert "limit_lowering_on_unknown_contact_state and not is_ventilation_enabled" in text

    def test_warns_without_a_configured_tilt_contact(self):
        text = self._checks()
        assert "limit_lowering_on_unknown_contact_state and is_ventilation_enabled and contact_window_tilted == []" in text
