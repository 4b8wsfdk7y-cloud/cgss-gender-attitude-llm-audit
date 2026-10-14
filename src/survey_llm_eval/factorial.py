"""Prompt construction for the 2 x 2 x 2 presentation experiment.

Factors
-------
A  battery visibility: only the target item is shown (0) or all items are
   listed before answering (1).
B  answer history: no earlier answers are in context (0) or items are
   answered in order and earlier answers remain in context (1).
C  output format: a bare integer (0) or a one-object JSON answer (1).

Cell A1-B1-C1 reproduces the joint five-item JSON condition of the v1 audit;
cell A0-B0-C1 reproduces its stateless single-item condition. Each function
returns chat messages plus the assistant text that precedes the option token,
so a caller can read the next-token distribution over the response scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

SYSTEM_PROMPT = "你正在参加一项社会调查模拟。严格遵守用户要求，不解释推理。"

INTRO = (
    "请阅读下述受访者资料，并以该受访者的身份作答。"
    "请选择最符合该受访者在调查年份可能看法的选项。"
    "没有正确答案，不需要解释理由。\n\n受访者资料：\n{persona}"
)

SCALE = (
    "评分标准：1=完全不同意；2=比较不同意；3=无所谓同意不同意；"
    "4=比较同意；5=完全同意。"
)

STEM = "请评价你对以下说法的同意程度："

CGSS_ITEMS: tuple[tuple[str, str], ...] = (
    ("A421", "男人以事业为重，女人以家庭为重。"),
    ("A422", "男性能力天生比女性强。"),
    ("A423", "干得好不如嫁得好。"),
    ("A424", "在经济不景气时，应该先解雇女性员工。"),
    ("A425", "夫妻应该均等分摊家务。"),
)


@dataclass(frozen=True)
class Cell:
    battery: bool
    history: bool
    json_format: bool

    @property
    def name(self) -> str:
        return f"A{int(self.battery)}-B{int(self.history)}-C{int(self.json_format)}"


CELLS: tuple[Cell, ...] = tuple(
    Cell(battery=a, history=b, json_format=c)
    for a in (False, True)
    for b in (False, True)
    for c in (False, True)
)


def cell_from_name(name: str) -> Cell:
    for cell in CELLS:
        if cell.name == name:
            return cell
    raise ValueError(f"Unknown cell {name!r}; expected one of {[c.name for c in CELLS]}")


def _battery_listing(items: Sequence[tuple[str, str]]) -> str:
    return "\n".join(f"{item_id}：{text}" for item_id, text in items)


def _single_format(item_id: str, json_format: bool) -> str:
    if json_format:
        return (
            f'只返回一个JSON对象，例如：{{"{item_id}": 1}}。'
            f"唯一的键必须为{item_id}，值必须为1至5的整数。不要输出任何其他文字。"
        )
    return "只回答一个1至5的整数，不要输出任何其他文字。"


def _sequential_format(items: Sequence[tuple[str, str]], json_format: bool) -> str:
    ids = "、".join(item_id for item_id, _ in items)
    if json_format:
        return (
            f"只返回一个JSON对象，键依次为{ids}，值必须为1至5的整数，"
            f'例如：{{"{items[0][0]}": 1, ...}}。不要输出任何其他文字。'
        )
    return (
        f"请依次回答{ids}，每行一个1至5的整数，共{len(items)}行。"
        "不要输出任何其他文字。"
    )


def _answer_text(item_id: str, answer: int, json_format: bool) -> str:
    return f'{{"{item_id}": {answer}}}' if json_format else str(answer)


def build_prompt(
    persona: str,
    target_index: int,
    cell: Cell,
    history: Sequence[int] = (),
    items: Sequence[tuple[str, str]] = CGSS_ITEMS,
) -> tuple[list[dict[str, str]], str]:
    """Return (messages, assistant_prefix) for one item in one cell.

    ``history`` holds the answers to items before ``target_index`` and is
    required exactly when the cell includes answer history.
    """
    if not 0 <= target_index < len(items):
        raise IndexError("target_index is outside the item battery")
    if cell.history and len(history) != target_index:
        raise ValueError("History cells need one earlier answer per preceding item")
    if not cell.history and history:
        raise ValueError("Cells without answer history must not receive history")

    item_id, item_text = items[target_index]
    intro = INTRO.format(persona=persona)
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if cell.battery and cell.history:
        user = "\n\n".join(
            [intro, STEM + "\n" + _battery_listing(items), SCALE,
             _sequential_format(items, cell.json_format)]
        )
        messages.append({"role": "user", "content": user})
        if cell.json_format:
            answered = ", ".join(
                f'"{items[i][0]}": {answer}' for i, answer in enumerate(history)
            )
            prefix = "{" + (answered + ", " if answered else "") + f'"{item_id}": '
        else:
            prefix = "".join(f"{answer}\n" for answer in history)
        return messages, prefix

    if cell.battery:
        user = "\n\n".join(
            [intro, STEM + "\n" + _battery_listing(items), SCALE,
             f"请只回答其中的{item_id}。" + _single_format(item_id, cell.json_format)]
        )
        messages.append({"role": "user", "content": user})
        return messages, _target_prefix(item_id, cell.json_format)

    if cell.history:
        for index in range(target_index + 1):
            current_id, current_text = items[index]
            question = "\n\n".join(
                [STEM + "\n" + f"{current_id}：{current_text}", SCALE,
                 _single_format(current_id, cell.json_format)]
            )
            content = intro + "\n\n" + question if index == 0 else question
            messages.append({"role": "user", "content": content})
            if index < target_index:
                messages.append(
                    {"role": "assistant",
                     "content": _answer_text(current_id, history[index], cell.json_format)}
                )
        return messages, _target_prefix(item_id, cell.json_format)

    user = "\n\n".join(
        [intro, STEM + "\n" + f"{item_id}：{item_text}", SCALE,
         _single_format(item_id, cell.json_format)]
    )
    messages.append({"role": "user", "content": user})
    return messages, _target_prefix(item_id, cell.json_format)


def _target_prefix(item_id: str, json_format: bool) -> str:
    return f'{{"{item_id}": ' if json_format else ""


def item_ids(items: Sequence[tuple[str, str]] = CGSS_ITEMS) -> tuple[str, ...]:
    return tuple(item_id for item_id, _ in items)


def history_answers(
    answers: Mapping[str, int], target_index: int,
    items: Sequence[tuple[str, str]] = CGSS_ITEMS,
) -> list[int]:
    """Earlier answers in battery order, for use as ``history``."""
    return [answers[items[i][0]] for i in range(target_index)]
