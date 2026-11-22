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


class TrustBatteryTest(unittest.TestCase):
    def test_trust_items_and_scale(self) -> None:
        from survey_llm_eval.factorial import BATTERIES, TRUST

        self.assertIs(BATTERIES["trust"], TRUST)
        self.assertEqual(TRUST.options, (1, 2, 3, 4))
        self.assertEqual(len(TRUST.items), 8)
        messages, prefix = build_prompt(
            PERSONA, 2, cell_from_name("A1-B1-C1"), history=[4, 3], battery=TRUST
        )
        text = user_text(messages)
        self.assertIn("1=完全不信任", text)
        self.assertIn("值必须为1至4的整数", text)
        self.assertNotIn("1至5", text)
        self.assertNotIn("A421", text)
        self.assertEqual(prefix, '{"N1601": 4, "N1603": 3, "N1605": ')

    def test_trust_single_item_format(self) -> None:
        from survey_llm_eval.factorial import TRUST

        messages, prefix = build_prompt(PERSONA, 0, cell_from_name("A0-B0-C0"), battery=TRUST)
        text = user_text(messages)
        self.assertIn("N1601：亲戚", text)
        self.assertNotIn("N1603", text)
        self.assertIn("只回答一个1至4的整数", text)
        self.assertEqual(prefix, "")


class IsspBatteryTest(unittest.TestCase):
    def test_issp_items_and_scale(self) -> None:
        from survey_llm_eval.factorial import BATTERIES, ISSP, STEM

        self.assertIs(BATTERIES["issp"], ISSP)
        self.assertEqual(ISSP.options, (1, 2, 3, 4, 5))
        self.assertEqual([i for i, _ in ISSP.items],
                         ["M101", "M102", "M103", "M104", "M105", "M201", "M202"])
        self.assertEqual(ISSP.stem, STEM)
        messages, prefix = build_prompt(
            PERSONA, 5, cell_from_name("A1-B1-C0"), history=[4, 2, 2, 3, 3], battery=ISSP
        )
        text = user_text(messages)
        self.assertIn("1=非常不同意", text)
        self.assertIn("M201：夫妻双方都应该挣钱养家。", text)
        self.assertNotIn("A421", text)
        self.assertEqual(prefix, "4\n2\n2\n3\n3\n")

    def test_issp_dialogue_history(self) -> None:
        from survey_llm_eval.factorial import ISSP

        messages, prefix = build_prompt(
            PERSONA, 1, cell_from_name("A0-B1-C1"), history=[5], battery=ISSP
        )
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[2]["content"], '{"M101": 5}')
        self.assertNotIn("M103", user_text(messages))
        self.assertEqual(prefix, '{"M102": ')


class WordingExperimentTest(unittest.TestCase):
    def test_variants_replace_only_their_items(self) -> None:
        from survey_llm_eval.factorial import (
            BATTERIES, GENDER, ISSP, TRUST, WORDING_VARIANTS, with_wording)

        for source in (GENDER, TRUST, ISSP):
            variant = BATTERIES[f"{source.name}_wording"]
            self.assertEqual(variant, with_wording(source))
            self.assertEqual((variant.stem, variant.scale, variant.labels),
                             (source.stem, source.scale, source.labels))
            changed = {i for (i, a), (_, b) in zip(source.items, variant.items) if a != b}
            self.assertEqual(changed, set(WORDING_VARIANTS[source.name]))
        self.assertEqual(sum(len(v) for v in WORDING_VARIANTS.values()), 5)

    def test_single_item_prompt_differs_only_in_the_item_text(self) -> None:
        from survey_llm_eval.factorial import BATTERIES, ISSP

        variant = BATTERIES["issp_wording"]
        for cell in ("A0-B0-C0", "A0-B0-C1"):
            old, old_prefix = build_prompt(PERSONA, 4, cell_from_name(cell), battery=ISSP)
            new, new_prefix = build_prompt(PERSONA, 4, cell_from_name(cell), battery=variant)
            self.assertEqual(old_prefix, new_prefix)
            self.assertEqual(
                user_text(old).replace(ISSP.items[4][1], variant.items[4][1]), user_text(new))
            self.assertIn("M105：对女人来说，当好家庭妇女也是一种成就。", user_text(new))
            self.assertNotIn("M201", user_text(new))

    def test_original_batteries_unchanged(self) -> None:
        from survey_llm_eval.factorial import GENDER, ISSP, TRUST

        self.assertEqual(GENDER.items[2], ("A423", "干得好不如嫁得好。"))
        self.assertEqual(TRUST.items[2], ("N1605", "陌生人"))
        self.assertEqual(ISSP.items[5], ("M201", "夫妻双方都应该挣钱养家。"))
