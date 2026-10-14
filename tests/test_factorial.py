import unittest

from survey_llm_eval.factorial import CELLS, build_prompt, cell_from_name

PERSONA = "调查年份：2018年；性别：女；年龄：45岁。"


def user_text(messages):
    return "\n".join(m["content"] for m in messages if m["role"] == "user")


class FactorialPromptTest(unittest.TestCase):
    def test_eight_distinct_cells(self) -> None:
        self.assertEqual(len(CELLS), 8)
        self.assertEqual(len({cell.name for cell in CELLS}), 8)
        self.assertEqual(cell_from_name("A1-B1-C1"), CELLS[-1])
        with self.assertRaises(ValueError):
            cell_from_name("A2-B0-C0")

    def test_battery_visibility(self) -> None:
        hidden, _ = build_prompt(PERSONA, 2, cell_from_name("A0-B0-C0"))
        shown, _ = build_prompt(PERSONA, 2, cell_from_name("A1-B0-C0"))
        self.assertNotIn("A421", user_text(hidden))
        self.assertIn("A423", user_text(hidden))
        for item_id in ("A421", "A422", "A423", "A424", "A425"):
            self.assertIn(item_id, user_text(shown))

    def test_joint_json_prefix_carries_history(self) -> None:
        messages, prefix = build_prompt(
            PERSONA, 2, cell_from_name("A1-B1-C1"), history=[4, 2]
        )
        self.assertEqual(prefix, '{"A421": 4, "A422": 2, "A423": ')
        self.assertEqual(len(messages), 2)

    def test_joint_digit_prefix(self) -> None:
        _, prefix = build_prompt(PERSONA, 2, cell_from_name("A1-B1-C0"), history=[4, 2])
        self.assertEqual(prefix, "4\n2\n")

    def test_dialogue_history_hides_later_items(self) -> None:
        messages, prefix = build_prompt(
            PERSONA, 2, cell_from_name("A0-B1-C0"), history=[4, 2]
        )
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual(messages[2]["content"], "4")
        self.assertNotIn("A424", user_text(messages))
        self.assertEqual(prefix, "")

    def test_first_item_needs_no_history(self) -> None:
        _, prefix = build_prompt(PERSONA, 0, cell_from_name("A1-B1-C1"))
        self.assertEqual(prefix, '{"A421": ')

    def test_history_validation(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt(PERSONA, 2, cell_from_name("A0-B1-C0"), history=[4])
        with self.assertRaises(ValueError):
            build_prompt(PERSONA, 2, cell_from_name("A0-B0-C0"), history=[4, 2])


if __name__ == "__main__":
    unittest.main()
