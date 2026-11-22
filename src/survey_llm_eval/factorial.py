"""Prompt construction for the 2 x 2 x 2 presentation experiment.

Factors
-------
A  battery visibility: only the target item is shown (0) or all items are
   listed before answering (1).
B  answer history: no earlier answers are in context (0) or items are
   answered in order and earlier answers remain in context (1).
C  output format: a bare integer (0) or a one-object JSON answer (1).
   An exploratory third level (2) asks for a verbal label before each score.

Cells A1-B1-C1 and A0-B0-C1 share the battery visibility, answer order, and
integer JSON output of the v1 joint and stateless single-item conditions. They
differ from v1 in the system prompt, in the user prompt (v1 also asked for a
verbal label, which its integer-only response schema did not allow), and in
the chat template: v1 ran with thinking enabled, so its answers were generated
inside an open ``<think>`` block (see PREREGISTRATION_V2.md, 2026-10-27).
Level C2 asks for a verbal label before each score. Each function returns chat
messages plus the assistant text that precedes the answer, so a caller can
read the next-token distribution over the response scale (or, for C2, over
the five labels).
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

LABELS: tuple[str, ...] = ("完全不同意", "比较不同意", "无所谓同意不同意", "比较同意", "完全同意")

CGSS_ITEMS: tuple[tuple[str, str], ...] = (
    ("A421", "男人以事业为重，女人以家庭为重。"),
    ("A422", "男性能力天生比女性强。"),
    ("A423", "干得好不如嫁得好。"),
    ("A424", "在经济不景气时，应该先解雇女性员工。"),
    ("A425", "夫妻应该均等分摊家务。"),
)

# Trust replication (PREREGISTRATION_V2.md, 2026-11-17): eight of the sixteen
# CGSS 2012 trust items (N16), in questionnaire order. CGSS codes 1 = 非常信任
# to 4 = 完全不信任; the prompt scale runs the other way, so higher scores mean
# more trust, as higher scores mean more agreement in the gender battery.
TRUST_ITEMS: tuple[tuple[str, str], ...] = (
    ("N1601", "亲戚"),
    ("N1603", "邻居"),
    ("N1605", "陌生人"),
    ("N1606", "医生"),
    ("N1611", "老师"),
    ("N1612", "地方政府官员"),
    ("N1613", "中央政府官员"),
    ("N1614", "警察"),
)
TRUST_STEM = "请评价你对以下这些人的信任程度："
TRUST_SCALE = "评分标准：1=完全不信任；2=不太信任；3=比较信任；4=非常信任。"
TRUST_LABELS: tuple[str, ...] = ("完全不信任", "不太信任", "比较信任", "非常信任")

# Second gender battery (PREREGISTRATION_V2.md, 2026-11-18): the seven
# agree-disagree gender-role items of the ISSP 2012 Family and Changing Gender
# Roles module as fielded in CGSS 2012 (section M), in questionnaire order and
# with the CGSS wording. CGSS codes 1 = 非常同意 to 5 = 非常不同意; the prompt
# scale runs the other way, so higher scores mean more agreement, as in the
# first gender battery.
ISSP_ITEMS: tuple[tuple[str, str], ...] = (
    ("M101", "和不工作的妈妈一样，那些有工作的妈妈也能够和自己的小孩关系亲密、稳定。"),
    ("M102", "如果小孩还没上学，妈妈出去工作会影响小孩的成长。"),
    ("M103", "总的来说，如果女的有全职工作，家庭生活就会受影响。"),
    ("M104", "有工作当然好，但是大多数女人真正想要的是家庭和孩子。"),
    ("M105", "对女人来说，做一个家庭妇女和出去工作会有一样的成就感。"),
    ("M201", "夫妻双方都应该挣钱养家。"),
    ("M202", "男人应该负责挣钱，女人应该负责顾家。"),
)
ISSP_SCALE = (
    "评分标准：1=非常不同意；2=不同意；3=无所谓同意不同意；4=同意；5=非常同意。"
)
ISSP_LABELS: tuple[str, ...] = ("非常不同意", "不同意", "无所谓同意不同意", "同意", "非常同意")

_COUNT_WORDS = {4: "四", 5: "五"}


@dataclass(frozen=True)
class Battery:
    """An item battery with its question stem and response scale."""

    name: str
    items: tuple[tuple[str, str], ...]
    stem: str
    scale: str
    labels: tuple[str, ...]

    @property
    def n_options(self) -> int:
        return len(self.labels)

    @property
    def options(self) -> tuple[int, ...]:
        return tuple(range(1, self.n_options + 1))


GENDER = Battery("gender", CGSS_ITEMS, STEM, SCALE, LABELS)
TRUST = Battery("trust", TRUST_ITEMS, TRUST_STEM, TRUST_SCALE, TRUST_LABELS)
ISSP = Battery("issp", ISSP_ITEMS, STEM, ISSP_SCALE, ISSP_LABELS)

# Wording experiment (PREREGISTRATION_V2.md, 2026-11-22): five items reworded,
# three changing a surface cue and two near-synonymous controls. Each variant
# battery is its source battery with only these item texts replaced; it is
# scored in the cells without history and battery listing (A0-B0), where the
# prompt holds the target item alone.
WORDING_VARIANTS: Mapping[str, Mapping[str, str]] = {
    "gender": {
        "A422": "男人的能力生来就比女人强。",
        "A423": "对女人来说，嫁个好丈夫比自己工作干得好更重要。",
    },
    "trust": {"N1605": "不认识的人"},
    "issp": {
        "M105": "对女人来说，当好家庭妇女也是一种成就。",
        "M201": "一个家光靠一个人挣钱不够，夫妻都要工作。",
    },
}


def with_wording(battery: Battery) -> Battery:
    """The battery with the wording-experiment texts substituted."""
    variants = WORDING_VARIANTS[battery.name]
    items = tuple((i, variants.get(i, text)) for i, text in battery.items)
    return Battery(f"{battery.name}_wording", items, battery.stem, battery.scale, battery.labels)


BATTERIES: dict[str, Battery] = {
    b.name: b for b in (GENDER, TRUST, ISSP, *(with_wording(x) for x in (GENDER, TRUST, ISSP)))
}


@dataclass(frozen=True)
class Cell:
    battery: bool
    history: bool
    json_format: bool
    labels: bool = False

    def __post_init__(self) -> None:
        if self.labels and not self.json_format:
            raise ValueError("Label answers are defined only for JSON output")

    @property
    def name(self) -> str:
        output = 2 if self.labels else int(self.json_format)
        return f"A{int(self.battery)}-B{int(self.history)}-C{output}"


CELLS: tuple[Cell, ...] = tuple(
    Cell(battery=a, history=b, json_format=c)
    for a in (False, True)
    for b in (False, True)
    for c in (False, True)
)


# Added after Parts 1-2 as an exploratory cell (PREREGISTRATION_V2.md, Changes):
# the joint condition with the v1 label-then-score answer format.
EXPLORATORY_CELLS: tuple[Cell, ...] = (
    Cell(battery=True, history=True, json_format=True, labels=True),
)


def cell_from_name(name: str) -> Cell:
    known = CELLS + EXPLORATORY_CELLS
    for cell in known:
        if cell.name == name:
            return cell
    raise ValueError(f"Unknown cell {name!r}; expected one of {[c.name for c in known]}")


def _battery_listing(items: Sequence[tuple[str, str]]) -> str:
    return "\n".join(f"{item_id}：{text}" for item_id, text in items)


def _single_format(
    item_id: str, json_format: bool, labels: bool = False, battery: Battery = GENDER
) -> str:
    n = battery.n_options
    if labels:
        return (
            f'只返回一个JSON对象，并同时返回文字标签和对应数字，例如：'
            f'{{"{item_id}":{{"label":"{battery.labels[0]}","score":1}}}}。唯一的键必须为{item_id}。'
            f"label只能使用评分标准中的{_COUNT_WORDS[n]}个文字标签，"
            f"score必须是与label严格对应的1至{n}整数。"
            "不要输出任何其他文字。"
        )
    if json_format:
        return (
            f'只返回一个JSON对象，例如：{{"{item_id}": 1}}。'
            f"唯一的键必须为{item_id}，值必须为1至{n}的整数。不要输出任何其他文字。"
        )
    return f"只回答一个1至{n}的整数，不要输出任何其他文字。"


def _sequential_format(
    items: Sequence[tuple[str, str]], json_format: bool, labels: bool = False,
    battery: Battery = GENDER,
) -> str:
    n = battery.n_options
    ids = "、".join(item_id for item_id, _ in items)
    if labels:
        return (
            f"只返回一个JSON对象，键依次为{ids}。每个题项必须同时返回文字标签和对应数字，"
            f'例如：{{"{items[0][0]}":{{"label":"{battery.labels[0]}","score":1}},...}}。'
            f"label只能使用评分标准中的{_COUNT_WORDS[n]}个文字标签，"
            f"score必须是与label严格对应的1至{n}整数。"
            "不要输出任何其他文字。"
        )
    if json_format:
        return (
            f"只返回一个JSON对象，键依次为{ids}，值必须为1至{n}的整数，"
            f'例如：{{"{items[0][0]}": 1, ...}}。不要输出任何其他文字。'
        )
    return (
        f"请依次回答{ids}，每行一个1至{n}的整数，共{len(items)}行。"
        "不要输出任何其他文字。"
    )


def _labelled(item_id: str, answer: int, battery: Battery = GENDER) -> str:
    return f'"{item_id}":{{"label":"{battery.labels[answer - 1]}","score":{answer}}}'


def _answer_text(
    item_id: str, answer: int, json_format: bool, labels: bool = False,
    battery: Battery = GENDER,
) -> str:
    if labels:
        return "{" + _labelled(item_id, answer, battery) + "}"
    return f'{{"{item_id}": {answer}}}' if json_format else str(answer)


def build_prompt(
    persona: str,
    target_index: int,
    cell: Cell,
    history: Sequence[int] = (),
    items: Sequence[tuple[str, str]] | None = None,
    battery: Battery = GENDER,
) -> tuple[list[dict[str, str]], str]:
    """Return (messages, assistant_prefix) for one item in one cell.

    ``history`` holds the answers to items before ``target_index`` and is
    required exactly when the cell includes answer history. ``battery`` sets
    the items, stem and response scale; ``items`` overrides the battery's items.
    """
    if items is None:
        items = battery.items
    stem, scale = battery.stem, battery.scale
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
            [intro, stem + "\n" + _battery_listing(items), scale,
             _sequential_format(items, cell.json_format, cell.labels, battery)]
        )
        messages.append({"role": "user", "content": user})
        if cell.labels:
            answered = ",".join(
                _labelled(items[i][0], answer, battery) for i, answer in enumerate(history)
            )
            prefix = "{" + (answered + "," if answered else "") + f'"{item_id}":{{"label":"'
        elif cell.json_format:
            answered = ", ".join(
                f'"{items[i][0]}": {answer}' for i, answer in enumerate(history)
            )
            prefix = "{" + (answered + ", " if answered else "") + f'"{item_id}": '
        else:
            prefix = "".join(f"{answer}\n" for answer in history)
        return messages, prefix

    if cell.battery:
        user = "\n\n".join(
            [intro, stem + "\n" + _battery_listing(items), scale,
             f"请只回答其中的{item_id}。" + _single_format(item_id, cell.json_format, cell.labels, battery)]
        )
        messages.append({"role": "user", "content": user})
        return messages, _target_prefix(item_id, cell.json_format, cell.labels)

    if cell.history:
        for index in range(target_index + 1):
            current_id, current_text = items[index]
            question = "\n\n".join(
                [stem + "\n" + f"{current_id}：{current_text}", scale,
                 _single_format(current_id, cell.json_format, cell.labels, battery)]
            )
            content = intro + "\n\n" + question if index == 0 else question
            messages.append({"role": "user", "content": content})
            if index < target_index:
                messages.append(
                    {"role": "assistant",
                     "content": _answer_text(current_id, history[index], cell.json_format,
                                             cell.labels, battery)}
                )
        return messages, _target_prefix(item_id, cell.json_format, cell.labels)

    user = "\n\n".join(
        [intro, stem + "\n" + f"{item_id}：{item_text}", scale,
         _single_format(item_id, cell.json_format, cell.labels, battery)]
    )
    messages.append({"role": "user", "content": user})
    return messages, _target_prefix(item_id, cell.json_format, cell.labels)


def _target_prefix(item_id: str, json_format: bool, labels: bool = False) -> str:
    if labels:
        return f'{{"{item_id}":{{"label":"'
    return f'{{"{item_id}": ' if json_format else ""


def label_continuations(battery: Battery = GENDER) -> tuple[tuple[int, str], ...]:
    """Text completing each answer after a C2 prefix.

    The closing quote is left out: the tokenizer merges it with the following
    comma, so a bare quote token is improbable after any label. Scoring the
    label tokens alone gives a proper distribution as long as no label's
    tokens are a prefix of another's, which the scorer checks.
    """
    return tuple(enumerate(battery.labels, start=1))


def item_ids(items: Sequence[tuple[str, str]] = CGSS_ITEMS) -> tuple[str, ...]:
    return tuple(item_id for item_id, _ in items)


def history_answers(
    answers: Mapping[str, int], target_index: int,
    items: Sequence[tuple[str, str]] = CGSS_ITEMS,
) -> list[int]:
    """Earlier answers in battery order, for use as ``history``."""
    return [answers[items[i][0]] for i in range(target_index)]
