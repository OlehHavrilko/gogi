"""Машина состояний диалогового цикла."""

import pytest

from state import InvalidTransition, State, StateMachine


def test_starts_idle():
    assert StateMachine().state is State.IDLE


def test_happy_path_transitions():
    fsm = StateMachine()
    for target in (
        State.LISTENING,
        State.TRANSCRIBING,
        State.THINKING,
        State.EXECUTING_TOOL,
        State.THINKING,
        State.SPEAKING,
        State.IDLE,
    ):
        fsm.to(target)
        assert fsm.state is target


def test_invalid_transition_raises_and_keeps_state():
    fsm = StateMachine()
    fsm.to(State.THINKING)
    with pytest.raises(InvalidTransition):
        fsm.to(State.LISTENING)  # THINKING -> LISTENING не разрешён
    assert fsm.state is State.THINKING


def test_same_state_is_noop_and_does_not_fire_callback():
    seen = []
    fsm = StateMachine(on_change=lambda old, new: seen.append((old, new)))
    fsm.to(State.THINKING)
    fsm.to(State.THINKING)
    assert seen == [(State.IDLE, State.THINKING)]


def test_on_change_receives_old_and_new():
    seen = []
    fsm = StateMachine(on_change=lambda old, new: seen.append((old, new)))
    fsm.to(State.LISTENING)
    fsm.to(State.TRANSCRIBING)
    assert seen == [
        (State.IDLE, State.LISTENING),
        (State.LISTENING, State.TRANSCRIBING),
    ]


def test_interrupt_jumps_to_listening_from_any_working_state():
    fsm = StateMachine()
    fsm.to(State.THINKING)
    fsm.to(State.SPEAKING)
    fsm.interrupt()
    assert fsm.state is State.LISTENING


def test_interrupt_to_idle():
    fsm = StateMachine()
    fsm.to(State.THINKING)
    fsm.interrupt(to=State.IDLE)
    assert fsm.state is State.IDLE


def test_interrupt_rejects_other_targets():
    with pytest.raises(ValueError):
        StateMachine().interrupt(to=State.SPEAKING)


def test_reset_returns_to_idle():
    fsm = StateMachine()
    fsm.to(State.THINKING)
    fsm.reset()
    assert fsm.state is State.IDLE


def test_error_state_can_only_recover_to_idle_or_listening():
    fsm = StateMachine()
    fsm.to(State.THINKING)
    fsm.to(State.ERROR)
    with pytest.raises(InvalidTransition):
        fsm.to(State.SPEAKING)
    fsm.to(State.IDLE)
    assert fsm.state is State.IDLE
