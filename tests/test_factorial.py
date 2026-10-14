import unittest

from survey_llm_eval.factorial import (
    CELLS,
    EXPLORATORY_CELLS,
    Cell,
    build_prompt,
    cell_from_name,
    label_continuations,
)

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


    def test_label_cell_is_exploratory(self) -> None:
        cell = cell_from_name("A1-B1-C2")
        self.assertTrue(cell.labels)
        self.assertNotIn(cell, CELLS)
        self.assertEqual(EXPLORATORY_CELLS, (cell,))
        with self.assertRaises(ValueError):
            Cell(battery=True, history=True, json_format=False, labels=True)

    def test_label_prefix_carries_labelled_history(self) -> None:
        messages, prefix = build_prompt(
            PERSONA, 2, cell_from_name("A1-B1-C2"), history=[4, 2]
        )
        self.assertEqual(
            prefix,
            '{"A421":{"label":"比较同意","score":4},'
            '"A422":{"label":"比较不同意","score":2},"A423":{"label":"',
        )
        self.assertIn('"label"', user_text(messages))
        _, first = build_prompt(PERSONA, 0, cell_from_name("A1-B1-C2"))
        self.assertEqual(first, '{"A421":{"label":"')

    def test_label_continuations(self) -> None:
        continuations = dict(label_continuations())
        self.assertEqual(sorted(continuations), [1, 2, 3, 4, 5])
        self.assertEqual(continuations[3], "无所谓同意不同意")

if __name__ == "__main__":
    unittest.main()
