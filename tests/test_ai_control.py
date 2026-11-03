"""AI/keyboard control tests (stdlib only, no pygame needed)."""
import unittest

from game import ai_control


class TestAIControl(unittest.TestCase):
    def setUp(self):
        ai_control.clear_action()

    def tearDown(self):
        ai_control.clear_action()

    def test_command_mapping(self):
        self.assertEqual(ai_control.map_command_to_action("LEFT")["move"], -1.0)
        self.assertEqual(ai_control.map_command_to_action("RIGHT")["move"], 1.0)
        self.assertTrue(ai_control.map_command_to_action("UP")["jump"])
        self.assertTrue(ai_control.map_command_to_action("DOWN")["place"])
        self.assertTrue(
            ai_control.map_command_to_action("INTERACT")["attack"])
        neutral = ai_control.map_command_to_action("BOGUS")
        self.assertEqual(neutral["move"], 0.0)
        self.assertFalse(neutral["jump"])

    def test_merge_simultaneous(self):
        # Both sources drive at once: moves sum (clamped), buttons OR.
        merged = ai_control.merge_with_keyboard(
            {"move": 1.0, "jump": False},
            {"move": 1.0, "jump": True, "attack": True, "place": False,
             "torch": False, "craft": False, "slot": None},
        )
        self.assertEqual(merged["move"], 1.0)
        self.assertTrue(merged["jump"])
        self.assertTrue(merged["attack"])

        merged = ai_control.merge_with_keyboard(
            {"move": 0.0, "jump": False},
            {"move": -1.0, "jump": False, "attack": False, "place": False,
             "torch": False, "craft": False, "slot": 3},
        )
        self.assertEqual(merged["move"], -1.0)
        self.assertEqual(merged["slot"], 3)

    def test_keyboard_slot_wins_over_ai(self):
        merged = ai_control.merge_with_keyboard(
            {"move": 0.0, "slot": 1},
            {"move": 0.0, "jump": False, "attack": False, "place": False,
             "torch": False, "craft": False, "slot": 5},
        )
        self.assertEqual(merged["slot"], 1)

    def test_in_memory_roundtrip(self):
        ai_control.set_action(move=-1.0, jump=True)
        state = ai_control.get_action()
        self.assertEqual(state["move"], -1.0)
        self.assertTrue(state["jump"])
        ai_control.set_action_from_command("RIGHT")
        self.assertEqual(ai_control.get_action()["move"], 1.0)

    def test_mapping_from_dict(self):
        action = ai_control._action_from_mapping(
            {"command": "LEFT", "jump": True, "slot": 2})
        self.assertEqual(action["move"], -1.0)
        self.assertTrue(action["jump"])
        self.assertEqual(action["slot"], 2)

        held = ai_control._action_from_mapping({"left": True, "up": True})
        self.assertEqual(held["move"], -1.0)
        self.assertTrue(held["jump"])

    def test_eeg_controller_gate(self):
        ctrl = ai_control.EEGModelController(confidence=0.60, cooldown=0.0)
        # EMA starts uniform, so the first peaked frame still gates.
        first = ctrl.action_from_probs([0.02, 0.02, 0.9, 0.03, 0.03])
        self.assertEqual(first["move"], 0.0)
        action = first
        for _ in range(6):
            action = ctrl.action_from_probs([0.02, 0.02, 0.9, 0.03, 0.03])
        self.assertEqual(action["move"], -1.0)  # LEFT after smoothing

        ctrl2 = ai_control.EEGModelController(confidence=0.60, cooldown=0.0)
        flat = ctrl2.action_from_probs([0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertEqual(flat["move"], 0.0)
        self.assertFalse(flat["jump"])


if __name__ == "__main__":
    unittest.main()
