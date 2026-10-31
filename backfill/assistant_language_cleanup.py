#!/usr/bin/env python3
"""Remove hidden attention-annotation language from SFT assistant responses.

The generation stages had access to human-attention annotations that are not
present in the final SFT prompt.  Generated traces can consequently refer to a
``fixation list``, ``human-attended regions``, or related artifacts that the
training example never supplies.  This module rewrites those references as
direct image inspection and removes ``Note:`` labels while preserving their
following prose.

Replacement wording is selected deterministically from small context-specific
pools.  The same sample ID and source text always produce identical output,
while different samples do not all learn one canned replacement phrase.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass


CLEANUP_VERSION = "assistant-attention-language-v3"
# Keep deterministic wording choices stable when cleanup validation evolves.
# Changing this namespace would intentionally reshuffle replacement wording.
REPLACEMENT_VARIATION_NAMESPACE = "assistant-attention-language-v2"


@dataclass(frozen=True)
class RewriteRule:
    """One ordered rewrite with a deterministic replacement pool."""

    name: str
    pattern: re.Pattern[str]
    replacements: Sequence[str] | Callable[[re.Match[str], str], Sequence[str]]


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE)


FIXATION_NOUN = (
    r"(?:(?:provided|available|listed|given)\s+)?"
    r"(?:human\s+)?fixation(?:\s+(?:points?|lists?|data|patterns?|regions?|"
    r"areas?|locations?|coordinates?|maps?|cues?|clusters?|zones?)|s)?"
    r"(?:\s+(?:provided|available|listed|given))?"
)


def _cluster_replacements(match: re.Match[str], _: str) -> Sequence[str]:
    prep = match.group("prep").lower()
    return (
        f"the relevant visual details are concentrated {prep}",
        f"the strongest visible cues appear {prep}",
        f"the key image evidence is concentrated {prep}",
        f"the most informative parts of the image lie {prep}",
        f"the visually relevant content appears {prep}",
    )


def _attention_focus_replacements(
    match: re.Match[str], _: str
) -> Sequence[str]:
    prep = match.group("prep").lower()
    return (
        f"the relevant visual evidence appears {prep}",
        f"the most informative image content lies {prep}",
        f"the key visible details are concentrated {prep}",
        f"the strongest image cues occur {prep}",
        f"the pertinent part of the image is {prep}",
    )


def _soft_guide_replacements(match: re.Match[str], _: str) -> Sequence[str]:
    if match.group(0).split()[0].casefold().endswith("ing"):
        return (
            "inspecting the image directly",
            "reviewing the visible details",
            "using direct visual inspection",
            "examining the relevant image areas",
            "looking closely at the image",
        )
    return (
        "inspect the image directly",
        "review the visible details",
        "use direct visual inspection",
        "examine the relevant image areas",
        "look closely at the image",
    )


def _list_location_replacements(match: re.Match[str], _: str) -> Sequence[str]:
    if match.group("prep").casefold() == "from":
        return (
            "from direct visual inspection",
            "from the visible image evidence",
            "from the image content",
            "from the pertinent visual details",
            "from close inspection of the image",
        )
    return (
        "in the image",
        "among the visible details",
        "within the relevant image areas",
        "within the visible image content",
        "among the pertinent image details",
    )


def _fixating_replacements(match: re.Match[str], _: str) -> Sequence[str]:
    prep = match.group("prep").casefold()
    if prep == "on":
        return (
            "inspecting",
            "examining",
            "looking closely at",
            "reviewing",
            "checking",
        )
    return (
        f"inspecting the area {prep}",
        f"examining the region {prep}",
        f"looking closely {prep}",
        f"reviewing the image content {prep}",
        f"checking the visible details {prep}",
    )


def _subject_fixated_replacements(
    match: re.Match[str], _: str
) -> Sequence[str]:
    verb = match.group(0).casefold().split()[0]
    if verb == "are":
        return (
            "are also visible",
            "appear clearly",
            "are visually apparent",
            "are also identifiable",
            "are evident in the image",
        )
    if verb == "were":
        return (
            "were also visible",
            "appeared clearly",
            "were visually apparent",
            "were also identifiable",
            "were evident in the image",
        )
    if verb == "was":
        return (
            "was also visible",
            "appeared clearly",
            "was visually apparent",
            "was also identifiable",
            "was evident in the image",
        )
    return (
        "is also visible",
        "appears clearly",
        "is visually apparent",
        "is also identifiable",
        "is evident in the image",
    )


def _recorded_fixation_replacements(
    match: re.Match[str], _: str
) -> Sequence[str]:
    verb = match.group("verb").casefold()
    prep = (match.group("prep") or "").casefold()
    if verb == "were recorded":
        if prep == "during":
            return (
                "direct visual inspection took place during",
                "the image was examined during",
                "the relevant image areas were reviewed during",
            )
        if prep == "in":
            return (
                "the relevant image coordinates are in",
                "the visible locations are in",
                "the pertinent image positions appear in",
            )
        return (
            f"the relevant image coordinates fall {prep}",
            f"the visible locations appear {prep}",
            f"the pertinent image positions lie {prep}",
        )
    if prep == "during":
        return (
            "direct visual inspection took place during",
            "the image was examined during",
            "the relevant image areas were reviewed during",
        )
    if prep == "on":
        return (
            "visual inspection also covered",
            "the examined image areas also included",
            "direct review of the image also covered",
        )
    if prep:
        return (
            f"visual inspection also covered areas {prep}",
            f"the examined image areas extend {prep}",
            f"direct review also included locations {prep}",
        )
    return (
        "visual inspection took place",
        "the image was examined",
        "the relevant areas were reviewed",
    )


def _drawn_fixation_location_replacements(
    match: re.Match[str], _: str
) -> Sequence[str]:
    location = match.group("location") or ""
    target = match.group("target")
    return (
        f"the image location{location} falls within {target}",
        f"the point{location} lies in {target}",
        f"the visible position{location} is part of {target}",
        f"the image detail{location} appears inside {target}",
    )


def _annotation_region_replacements(
    match: re.Match[str], _: str
) -> Sequence[str]:
    noun = match.group("noun").casefold()
    if noun.endswith("s"):
        return (
            "visual cues",
            "relevant image areas",
            "visible details",
            "image locations",
            "points of interest",
            "pertinent visual regions",
        )
    return (
        "visual cue",
        "relevant image area",
        "visible detail",
        "image location",
        "point of interest",
        "pertinent visual region",
    )


def _human_visual_artifact_replacements(
    match: re.Match[str], _: str
) -> Sequence[str]:
    noun = match.group("noun").casefold()
    if noun == "cues":
        return (
            "visible cues",
            "relevant image areas",
            "visible details",
            "image locations",
            "points of interest",
            "pertinent visual regions",
        )
    return (
        "visible evidence",
        "relevant image content",
        "visual emphasis",
        "direct visual inspection",
        "pertinent visual detail",
    )


GENERATED_FINITE_LOCATION_PHRASE = (
    r"(?:the\s+image\s+contains\s+several\s+relevant\s+locations|"
    r"several\s+pertinent\s+image\s+regions\s+stand\s+out|"
    r"the\s+visible\s+content\s+contains\s+useful\s+points\s+of\s+reference|"
    r"several\s+relevant\s+coordinates\s+appear\s+in\s+the\s+image|"
    r"multiple\s+useful\s+image\s+locations\s+can\s+be\s+inspected)"
)


def _generated_location_context_replacements(
    match: re.Match[str], _: str
) -> Sequence[str]:
    prefix = match.group("prefix").casefold()
    if prefix == "in":
        return (
            "among the relevant image locations",
            "within the pertinent image areas",
            "among the visible points of reference",
            "within the useful image regions",
        )
    return (
        f"{prefix} the relevant image locations",
        f"{prefix} the pertinent visual areas",
        f"{prefix} the visible points of reference",
        f"{prefix} the useful image regions",
    )


def _generated_location_predicate_replacements(
    match: re.Match[str], _: str
) -> Sequence[str]:
    predicate = re.sub(r"\s+", " ", match.group("predicate").casefold())
    predicate = {
        "includes": "include",
        "contains": "contain",
        "shows": "show",
        "is": "are",
        "appears": "appear",
        "seems": "seem",
        "has": "have",
    }.get(predicate, predicate)
    return (
        f"the relevant image locations {predicate}",
        f"the pertinent visual areas {predicate}",
        f"the visible points of reference {predicate}",
        f"the useful image regions {predicate}",
    )


RULES: tuple[RewriteRule, ...] = (
    RewriteRule(
        "since_user_provided_fixations",
        _compile(
            rf"\bsince\s+(?:the\s+)?user\s+(?:has\s+)?provided\s+"
            rf"(?:the\s+)?{FIXATION_NOUN}\s*,\s*and\s+some\s+fall\b"
        ),
        (
            "since some relevant image locations fall",
            "since some pertinent visual points fall",
            "since several useful image coordinates fall",
            "since some visually relevant areas fall",
            "since a few informative image locations fall",
        ),
    ),
    RewriteRule(
        "user_provided_fixations",
        _compile(
            rf"\b(?:(?:the\s+)?user\s+(?:has\s+)?provided\s+(?:the\s+)?"
            rf"{FIXATION_NOUN}|user-provided\s+{FIXATION_NOUN})\b"
        ),
        (
            "the image contains several relevant locations",
            "several pertinent image regions stand out",
            "the visible content contains useful points of reference",
            "several relevant coordinates appear in the image",
            "multiple useful image locations can be inspected",
        ),
    ),
    RewriteRule(
        "fixations_or_regions",
        _compile(rf"\b{FIXATION_NOUN}\s+or\s+(?:image\s+)?regions?\b"),
        (
            "image regions",
            "visible areas",
            "relevant image locations",
            "pertinent visual areas",
            "parts of the image",
        ),
    ),
    RewriteRule(
        "area_around_fixated_region",
        _compile(
            r"\b(?:the\s+)?area\s+around\s+(?:the\s+)?"
            r"(?:(?:human[- ]|heavily\s+|strongly\s+)?fixated)\s+"
            r"region\s+near\b"
        ),
        (
            "the image area near",
            "the visual region near",
            "the part of the image near",
            "the visible area near",
            "the image section near",
        ),
    ),
    RewriteRule(
        "described_by_fixations_and_context",
        _compile(
            rf"\bas\s+described\s+by\s+(?:the\s+)?{FIXATION_NOUN}\s+and\s+"
            rf"(?P<context>(?:standard|typical)\s+(?:interpretation|conventions?))\b"
        ),
        lambda match, _: (
            f"based on direct image inspection and {match.group('context')}",
            f"based on the visible content and {match.group('context')}",
            f"using the image itself and {match.group('context')}",
        ),
    ),
    RewriteRule(
        "high_attention_regions",
        _compile(
            r"\bhigh[- ]attention\s+"
            r"(?P<noun>points?|patterns?|regions?|areas?|locations?|"
            r"coordinates?|cues?|clusters?|zones?)\b"
        ),
        (
            "visually prominent areas",
            "key image regions",
            "salient visual locations",
            "important parts of the image",
            "highly relevant image areas",
        ),
    ),
    RewriteRule(
        "recorded_fixation_event",
        _compile(
            r"\b(?:(?:some|the|these|those|several|many)\s+)?"
            r"(?:human\s+)?fixations?\s+"
            r"(?P<verb>occurred|happened|were\s+recorded|were\s+observed)"
            r"(?:\s+(?P<prep>during|near|around|on|in|at))?\b"
        ),
        _recorded_fixation_replacements,
    ),
    RewriteRule(
        "drawn_fixation_location",
        _compile(
            r"\b(?:a\s+)?(?:human\s+)?fixation(?:\s+point)?"
            r"(?P<location>\s+(?:at|near|around)\s+(?:\*{0,2})?"
            r"\([^\n)]+\)(?:\*{0,2})?)\s+"
            r"(?:may|might|could)\s+have\s+been\s+drawn\s+"
            r"(?:to|toward|towards|on)\s+"
            r"(?P<target>(?:this|that|the)\s+"
            r"(?:region|area|location|section|part)(?:\s+of\s+the\s+image)?)\b"
        ),
        _drawn_fixation_location_replacements,
    ),
    RewriteRule(
        "article_fixation_singular",
        _compile(
            r"\b(?:a|an)\s+(?:(?:provided|available|listed|given)\s+)?"
            r"(?:human\s+)?fixation(?:\s+point)?"
            r"(?:\s+(?:provided|available|listed|given))?\b"
        ),
        (
            "a visual cue",
            "an image detail",
            "a relevant image location",
            "a point of interest",
            "a salient visual region",
            "an examined image area",
        ),
    ),
    RewriteRule(
        "coordinates_provided_in_annotations",
        _compile(
            rf"\b(?:coordinates?|points?|locations?)\s+provided\s+"
            rf"(?:in|by|through)\s+(?:the\s+)?{FIXATION_NOUN}\b"
        ),
        (
            "relevant image coordinates",
            "specific points in the image",
            "visible image locations",
            "pertinent visual coordinates",
            "referenced image points",
        ),
    ),
    RewriteRule(
        "attention_distribution",
        _compile(
            r"\b(?:(?:spatial|overall|observed|dense)\s+)?"
            r"(?:distribution|density|concentration|clustering)\s+of\s+"
            r"(?:(?:human|visual)\s+)?attention\b"
        ),
        (
            "spatial arrangement of the visual cues",
            "distribution of the visible details",
            "concentration of relevant image content",
            "pattern formed by the image locations",
            "spatial layout of the pertinent visual evidence",
        ),
    ),
    RewriteRule(
        "soft_guide",
        _compile(
            rf"\b(?:use|using|review|reviewing|consider|considering|treat|treating)\s+"
            rf"(?:the\s+)?{FIXATION_NOUN}\s+as\s+(?:a\s+)?"
            rf"(?:soft\s+)?(?:guides?|hints?|priors?)\b"
        ),
        _soft_guide_replacements,
    ),
    RewriteRule(
        "annotation_confirms_attention",
        _compile(
            rf"\b(?:(?:the|these|those|such|provided|available|many|several)\s+)?"
            rf"{FIXATION_NOUN}\s+"
            rf"(?:(?:clearly|strongly|also|further)\s+)?"
            rf"(?:confirm|confirms|support|supports|suggest|suggests|"
            rf"indicate|indicates|show|shows|reinforce|reinforces|"
            rf"highlight|highlights)\s+(?:that\s+)?"
            rf"(?:human\s+visual\s+|visual\s+|human\s+)?attention\s+(?:was|is)\s+"
            rf"(?:paid|drawn|directed|focused)\s+"
            rf"(?:to|toward|towards|on)\b"
        ),
        (
            "the image highlights",
            "the visible evidence emphasizes",
            "direct inspection points to",
            "the image content underscores",
            "the visual details clearly show",
        ),
    ),
    RewriteRule(
        "combined_inspection_cluster",
        _compile(
            rf"\b(?:looking(?:\s+back)?\s+at|checking|reviewing|examining|"
            rf"re-?evaluating|cross-?checking)\s+(?:the\s+)?{FIXATION_NOUN}"
            rf"\s*:\s*(?:many|multiple|several|numerous)\s+(?:points|locations)\s+"
            rf"(?:are\s+|were\s+)?(?:densely\s+|heavily\s+|strongly\s+)?"
            rf"cluster(?:ed|ing)?\s+(?P<prep>around|near|in|on|at)\b"
        ),
        _cluster_replacements,
    ),
    RewriteRule(
        "lets_inspect",
        _compile(
            rf"\blet['’]s\s+(?:check|review|use|look\s+at|re-?examine|"
            rf"cross-?check)\s+(?:the\s+)?{FIXATION_NOUN}\b"
        ),
        (
            "let's inspect the image closely",
            "let's examine the visible details",
            "let's review the relevant image areas",
            "let's look carefully at the image",
            "let's verify this directly from the image",
        ),
    ),
    RewriteRule(
        "gerund_inspection",
        _compile(
            rf"\b(?:looking(?:\s+back)?\s+at|checking|reviewing|using|"
            rf"examining|re-?evaluating|cross-?checking)\s+"
            rf"(?:the\s+)?{FIXATION_NOUN}\b"
        ),
        (
            "inspecting the image closely",
            "examining the visible details",
            "reviewing the relevant image areas",
            "looking carefully at the image",
            "checking the image evidence directly",
        ),
    ),
    RewriteRule(
        "inspection_attribution",
        _compile(
            rf"\b(?:based\s+on|according\s+to|using|from)\s+"
            rf"(?:the\s+)?{FIXATION_NOUN}\b"
        ),
        (
            "from direct visual inspection",
            "based on the visible image evidence",
            "from the image content",
            "through close inspection of the image",
            "based on the visible details",
        ),
    ),
    RewriteRule(
        "given_visual_evidence",
        _compile(rf"\bgiven\s+(?:the\s+)?{FIXATION_NOUN}\b"),
        (
            "given the visible evidence",
            "considering the image content",
            "with the relevant visual details in view",
            "based on direct image inspection",
            "considering the visible cues",
        ),
    ),
    RewriteRule(
        "as_seen_in_image",
        _compile(
            rf"\bas\s+(?:indicated|shown|confirmed|supported|suggested)\s+by\s+"
            rf"(?:both\s+visual\s+inspection\s+and\s+)?"
            rf"(?:the\s+)?{FIXATION_NOUN}\b"
        ),
        (
            "as seen directly in the image",
            "as shown by the visible details",
            "as supported by close image inspection",
            "as the image itself indicates",
            "as the visual evidence shows",
        ),
    ),
    RewriteRule(
        "clustered_visual_details",
        _compile(
            rf"\b(?:there\s+(?:are|were)\s+)?"
            rf"(?:(?:many|multiple|several|numerous|most|some|the|these|those)\s+)?"
            rf"{FIXATION_NOUN}\s+"
            rf"(?:(?:are|were|seem(?:ed)?|appear(?:ed)?)\s+)?"
            rf"(?:(?:very|most|more|strongly|heavily|densely|tightly)\s+)?"
            rf"(?:cluster(?:ed|ing)?|concentrat(?:ed|ing))\s+"
            rf"(?P<prep>around|near|in|on|at)\b"
        ),
        _cluster_replacements,
    ),
    RewriteRule(
        "visual_evidence_supports",
        _compile(
            rf"\b(?:(?:the|these|those|such|provided|available|many|several)\s+)?"
            rf"{FIXATION_NOUN}\s+"
            rf"(?:(?:clearly|strongly|also|further)\s+)?"
            rf"(?:confirm|confirms|support|supports|suggest|suggests|"
            rf"indicate|indicates|show|shows|reinforce|reinforces|"
            rf"highlight|highlights)\b"
        ),
        (
            "the visual evidence supports",
            "the image content indicates",
            "the visible details reinforce",
            "direct inspection suggests",
            "the image itself shows",
            "the relevant visual cues support",
        ),
    ),
    RewriteRule(
        "attention_focus",
        _compile(
            r"(?<!my\s)(?<!our\s)(?<!your\s)\b"
            r"(?:the\s+)?(?:human\s+visual\s+|visual\s+|human\s+)?attention\s+"
            r"(?:(?:was|is|seems?|appears?)\s+)?"
            r"(?:(?:strongly|mainly|primarily|heavily)\s+)?"
            r"(?:drawn|directed|focused|concentrated|clustered|centred|centered)\s+"
            r"(?P<prep>around|near|in|on|at|toward|towards)\b"
        ),
        _attention_focus_replacements,
    ),
    RewriteRule(
        "relative_attention_focus",
        _compile(
            r"\bregions?\s+where\s+"
            r"(?:human\s+visual\s+|visual\s+|human\s+)?attention\s+"
            r"(?:was|is)\s+(?:paid|drawn|directed|focused)"
            r"(?:\s+(?:to|toward|towards|on))?\b"
        ),
        (
            "visually relevant regions",
            "pertinent image areas",
            "important parts of the image",
            "salient visual regions",
            "key image locations",
        ),
    ),
    RewriteRule(
        "bare_attention_focus_state",
        _compile(
            r"(?<!my\s)(?<!our\s)(?<!your\s)\b"
            r"(?:the\s+)?(?:human\s+visual\s+|visual\s+|human\s+)?attention\s+"
            r"(?:was|is|seems?\s+to\s+be|appears?\s+to\s+be)\s+"
            r"(?:focused|concentrated|clustered|centred|centered)\b"
        ),
        (
            "the relevant visual content appears",
            "the key image evidence lies",
            "the pertinent visible detail occurs",
            "the strongest image cue appears",
            "the visual emphasis falls",
        ),
    ),
    RewriteRule(
        "embedded_attention_emphasis",
        _compile(
            r"\b(?:confirming|indicating|suggesting|showing|reinforcing)\s+"
            r"(?:that\s+)?(?:human\s+visual\s+|visual\s+|human\s+)?attention\s+"
            r"(?:was|is)\s+(?:paid|drawn|directed|focused)\s+"
            r"(?:to|toward|towards|on)\b"
        ),
        (
            "highlighting",
            "emphasizing",
            "pointing to",
            "underscoring",
            "visually prioritizing",
        ),
    ),
    RewriteRule(
        "embedded_attention_emphasis_bare",
        _compile(
            r"\b(?:confirming|indicating|suggesting|showing|reinforcing)\s+"
            r"(?:that\s+)?(?:human\s+visual\s+|visual\s+|human\s+)?attention\s+"
            r"(?:was|is)\s+(?:paid|drawn|directed|focused)"
            r"(?:\s+(?:here|there))?\b"
        ),
        (
            "highlighting this area",
            "emphasizing that region",
            "pointing to this location",
            "underscoring this part of the image",
            "visually reinforcing this area",
        ),
    ),
    RewriteRule(
        "bare_attention_emphasis",
        _compile(
            r"(?<!my\s)(?<!our\s)(?<!your\s)\b"
            r"(?:the\s+)?(?:human\s+visual\s+|visual\s+|human\s+)?attention\s+"
            r"(?:was|is)\s+(?:paid|drawn|directed|focused)\s+"
            r"(?:to|toward|towards|on)\b"
        ),
        (
            "the image emphasizes",
            "the visual evidence highlights",
            "the visible content points to",
            "direct inspection centers on",
            "the image draws focus to",
        ),
    ),
    RewriteRule(
        "bare_attention_emphasis_without_object",
        _compile(
            r"(?<!my\s)(?<!our\s)(?<!your\s)\b"
            r"(?:the\s+)?(?:human\s+visual\s+|visual\s+|human\s+)?attention\s+"
            r"(?:was|is)\s+(?:paid|drawn|directed|focused)"
            r"(?:\s+(?:here|there))?\b"
        ),
        (
            "the image emphasizes this area",
            "the visual evidence highlights this region",
            "the visible content points to this location",
            "direct inspection centers on this area",
            "the image draws focus to this region",
        ),
    ),
    RewriteRule(
        "attention_state_location",
        _compile(
            r"(?<!my\s)(?<!our\s)(?<!your\s)\b"
            r"(?:the\s+)?(?:human\s+visual\s+|visual\s+|human\s+)?attention\s+"
            r"(?:was|is|seems?\s+to\s+be|appears?\s+to\s+be)\s+"
            r"(?P<prep>around|near|in|on|at|toward|towards)\b"
        ),
        _attention_focus_replacements,
    ),
    RewriteRule(
        "attention_clusters",
        _compile(
            r"(?<!my\s)(?<!our\s)(?<!your\s)\b"
            r"(?:the\s+)?(?:human\s+visual\s+|visual\s+|human\s+)?attention\s+"
            r"(?:clearly\s+|strongly\s+|heavily\s+|tightly\s+)?"
            r"(?:clusters?|centers?|centres?)\s+"
            r"(?P<prep>around|near|in|on|at)\b"
        ),
        _attention_focus_replacements,
    ),
    RewriteRule(
        "user_fixated_location",
        _compile(
            r"\b(?:the\s+)?(?:user|human|viewer|observer)\s+"
            r"(?:has\s+|had\s+)?fixat(?:ed|es)\s+"
            r"(?:(?:very|most|more|strongly|heavily|densely|tightly)\s+)?"
            r"(?:around|near|in|on|at)\b"
        ),
        (
            "the relevant visual detail appears at",
            "the pertinent image content is visible at",
            "the key image evidence occurs at",
            "the visually informative area is at",
            "the important image region lies at",
        ),
    ),
    RewriteRule(
        "possessive_user_gaze",
        _compile(
            r"\b(?:the\s+)?(?:user|human|viewer|observer)['’]s\s+gaze"
            r"(?:\s+history)?\b"
        ),
        (
            "direct visual inspection",
            "the visible evidence",
            "the image content",
            "the relevant visual details",
            "the image evidence",
        ),
    ),
    RewriteRule(
        "human_gaze",
        _compile(r"\b(?:the\s+)?(?:human|user)\s+gaze\b"),
        (
            "visual focus",
            "image emphasis",
            "the visible evidence",
            "the relevant image content",
            "visual inspection",
        ),
    ),
    RewriteRule(
        "gaze_annotation_plural",
        _compile(
            r"\bgaze\s+(?P<noun>points?|patterns?|regions?|areas?|locations?|"
            r"coordinates?|maps?|cues?|clusters?|zones?|fixations?)"
            r"(?:\s+(?:provided|available|listed|given))?\b"
        ),
        _annotation_region_replacements,
    ),
    RewriteRule(
        "gaze_annotation_data",
        _compile(r"\bgaze\s+(?:data|lists?)\b"),
        (
            "visual evidence",
            "image evidence",
            "visible information",
            "image content",
            "visual information",
        ),
    ),
    RewriteRule(
        "visual_attention_plural_artifact",
        _compile(
            r"\b(?:human\s+)?visual\s+attention\s+"
            r"(?P<noun>points?|patterns?|regions?|areas?|locations?|coordinates?|"
            r"maps?|cues?|clusters?|zones?|lists?|fixations?)"
            r"(?:\s+(?:provided|available|listed|given))?\b"
        ),
        _annotation_region_replacements,
    ),
    RewriteRule(
        "visual_attention_data",
        _compile(r"\b(?:human\s+)?visual\s+attention\s+data\b"),
        (
            "visual evidence",
            "image evidence",
            "visible information",
            "image content",
            "visual information",
        ),
    ),
    RewriteRule(
        "visual_attention_bare",
        _compile(r"\b(?:human\s+)?visual\s+attention\b"),
        (
            "visual emphasis",
            "image focus",
            "visible evidence",
            "relevant image content",
            "visual importance",
        ),
    ),
    RewriteRule(
        "regions_fixated_by_user",
        _compile(
            r"\b(?:(?:heavily|strongly|densely|closely)\s+)?"
            r"(?P<noun>regions?|areas?|zones?|locations?|points?)\s+"
            r"fixated\s+by\s+(?:the\s+)?(?:user|human|viewer|observer)\b"
        ),
        _annotation_region_replacements,
    ),
    RewriteRule(
        "fixated_regions",
        _compile(
            r"\b(?:(?:heavily|strongly|densely|closely)\s+|human[- ])?"
            r"fixated\s+(?P<noun>regions?|areas?|zones?|locations?|points?)\b"
        ),
        _annotation_region_replacements,
    ),
    RewriteRule(
        "subject_is_fixated",
        _compile(r"\b(?:is|are|was|were)\s+(?:also\s+)?fixated\b"),
        _subject_fixated_replacements,
    ),
    RewriteRule(
        "degree_of_fixation",
        _compile(
            r"\b(?:(?:the\s+)?most|more|heavily|strongly|densely|closely)\s+"
            r"fixated(?:\s+(?:upon|on))?\b"
        ),
        (
            "the most visually prominent",
            "particularly salient",
            "visually emphasized",
            "especially prominent in the image",
            "visually notable",
        ),
    ),
    RewriteRule(
        "fixating_location",
        _compile(r"\bfixating\s+(?P<prep>around|near|in|on|at)\b"),
        _fixating_replacements,
    ),
    RewriteRule(
        "fixated_location",
        _compile(r"\bfixated\s+(?P<prep>around|near|in|on|at)\b"),
        (
            "visible at",
            "located at",
            "apparent at",
            "identifiable at",
            "evident at",
        ),
    ),
    RewriteRule(
        "eye_movement",
        _compile(r"\b(?:transitional\s+)?eye\s+movements?\b"),
        (
            "surrounding visual context",
            "a peripheral image detail",
            "the surrounding scene",
            "nearby visual content",
            "an incidental image feature",
        ),
    ),
    RewriteRule(
        "human_attended_plural",
        _compile(
            r"\bhuman[- ]attended\s+"
            r"(?P<noun>regions?|areas?|points?|coordinates?|locations?|zones?|fixations?)"
            r"(?:\s+(?:provided|available|listed|given))?\b"
        ),
        _annotation_region_replacements,
    ),
    RewriteRule(
        "human_attended_bare",
        _compile(r"\bhuman[- ]attended\b"),
        (
            "visually relevant",
            "salient",
            "pertinent",
            "important in the image",
            "visually informative",
        ),
    ),
    RewriteRule(
        "human_attention_plural_artifact",
        _compile(
            r"\bhuman[- ]attention\s+"
            r"(?P<noun>points?|patterns?|regions?|areas?|locations?|coordinates?|"
            r"maps?|cues?|clusters?|zones?|lists?|fixations?)"
            r"(?:\s+(?:provided|available|listed|given))?\b"
        ),
        _annotation_region_replacements,
    ),
    RewriteRule(
        "human_attention_data",
        _compile(r"\bhuman[- ]attention\s+data\b"),
        (
            "visual evidence",
            "image evidence",
            "visible information",
            "image content",
            "visual information",
        ),
    ),
    RewriteRule(
        "human_attention_bare",
        _compile(r"\bhuman[- ]attention\b"),
        (
            "visual focus",
            "image focus",
            "focus within the image",
            "visual emphasis",
            "area of visual interest",
        ),
    ),
    RewriteRule(
        "fixation_plural_artifact",
        _compile(
            r"\b(?:(?:provided|available|listed|given)\s+)?"
            r"(?:human\s+)?fixation\s+"
            r"(?P<noun>patterns?|regions?|areas?|locations?|coordinates?|maps?|cues?|"
            r"clusters?|zones?)"
            r"(?:\s+(?:provided|available|listed|given))?\b"
        ),
        _annotation_region_replacements,
    ),
    RewriteRule(
        "fixation_distribution",
        _compile(
            r"\b(?:(?:spatial|overall|observed|dense)\s+)?fixation\s+"
            r"(?:density|distribution|concentration|clustering)\b"
        ),
        (
            "spatial arrangement of the visual cues",
            "distribution of the visible details",
            "concentration of relevant image content",
            "pattern formed by the image locations",
            "spatial layout of the pertinent visual evidence",
        ),
    ),
    RewriteRule(
        "fixation_list_location",
        _compile(
            r"\b(?P<prep>in|on|from|among)\s+(?:the\s+)?"
            r"(?:(?:provided|available|listed|given)\s+)?"
            r"(?:human\s+)?fixation\s+(?:lists?|data)\b"
        ),
        _list_location_replacements,
    ),
    RewriteRule(
        "one_of_fixation_locations",
        _compile(
            rf"\b(?:matches|matching|corresponds?\s+to|aligns?\s+with)\s+"
            rf"one\s+of\s+(?:the\s+)?{FIXATION_NOUN}\b"
        ),
        (
            "matches a visible image location",
            "corresponds to a relevant point in the image",
            "aligns with a pertinent visual area",
            "matches a specific image region",
            "corresponds to a visible coordinate",
        ),
    ),
    RewriteRule(
        "group_fixation_clusters",
        _compile(
            rf"\bgroup(?:ing)?\s+(?:the\s+)?{FIXATION_NOUN}\s+"
            rf"(?:in|into|by)\s+clusters?\b"
        ),
        (
            "grouping the visible elements by location",
            "organizing the relevant image areas spatially",
            "separating the visible subjects by location",
            "grouping the image content into distinct areas",
            "distinguishing the visible elements spatially",
        ),
    ),
    RewriteRule(
        "fixation_heading",
        _compile(rf"\b{FIXATION_NOUN}\s+(?:around|near|include)\s*:"),
        (
            "Relevant image regions include:",
            "Key visual areas include:",
            "Pertinent image locations include:",
            "Visible points of interest include:",
            "Important parts of the image include:",
        ),
    ),
    RewriteRule(
        "viewer_checked",
        _compile(r"\bwhere\s+the\s+(?:viewer|observer)\s+checked\s+for\b"),
        (
            "a region containing",
            "an area showing",
            "a location with",
            "part of the image containing",
            "an area that may show",
        ),
    ),
    RewriteRule(
        "peripheral_tracking",
        _compile(r"\b(?:peripheral|visual|eye)\s+tracking\b"),
        (
            "surrounding visual context",
            "a peripheral image detail",
            "the surrounding scene",
            "nearby visual content",
            "an incidental image feature",
        ),
    ),
    RewriteRule(
        "attention_plural_artifact",
        _compile(
            r"\battention\s+"
            r"(?P<noun>points?|patterns?|regions?|areas?|locations?|coordinates?|"
            r"cues?|clusters?|zones?|fixations?)"
            r"(?:\s+(?:provided|available|listed|given))?\b"
        ),
        _annotation_region_replacements,
    ),
    RewriteRule(
        "attention_map_or_data",
        _compile(r"\battention\s+(?:maps?|data|lists?)\b"),
        (
            "visual evidence",
            "image evidence",
            "visible information",
            "image content",
            "visual information",
        ),
    ),
    RewriteRule(
        "attended_regions",
        _compile(
            r"\battended\s+"
            r"(?P<noun>regions?|areas?|locations?|points?|coordinates?|zones?)"
            r"(?:\s+(?:provided|available|listed|given))?\b"
        ),
        _annotation_region_replacements,
    ),
    RewriteRule(
        "fixation_list_or_data",
        _compile(
            r"\b(?:(?:provided|available|listed|given)\s+)?"
            r"(?:human\s+)?fixation\s+(?:lists?|data)"
            r"(?:\s+(?:provided|available|listed|given))?\b"
        ),
        (
            "visual evidence",
            "image evidence",
            "visible information",
            "image content",
            "visual information",
        ),
    ),
    RewriteRule(
        "fixation_points_plural",
        _compile(
            r"\b(?:(?:provided|available|listed|given)\s+)?"
            r"(?:human\s+)?fixation\s+points"
            r"(?:\s+(?:provided|available|listed|given))?\b"
        ),
        (
            "image locations",
            "specific visual locations",
            "relevant coordinates",
            "points in the image",
            "visible positions",
            "key image areas",
        ),
    ),
    RewriteRule(
        "fixation_point_singular",
        _compile(
            r"\b(?:(?:provided|available|listed|given)\s+)?"
            r"(?:human\s+)?fixation\s+point"
            r"(?:\s+(?:provided|available|listed|given))?\b"
        ),
        (
            "image location",
            "specific visual location",
            "relevant coordinate",
            "point in the image",
            "visible position",
            "key image area",
        ),
    ),
    RewriteRule(
        "fixations_plural",
        _compile(
            r"\b(?:(?:provided|available|listed|given)\s+)?"
            r"(?:human\s+)?fixations"
            r"(?:\s+(?:provided|available|listed|given))?\b"
        ),
        (
            "visual cues",
            "image details",
            "relevant locations",
            "points of interest",
            "salient regions",
            "examined areas",
        ),
    ),
    RewriteRule(
        "fixation_singular",
        _compile(
            r"\b(?:(?:provided|available|listed|given)\s+)?"
            r"(?:human\s+)?fixation"
            r"(?:\s+(?:provided|available|listed|given))?\b"
        ),
        (
            "visual cue",
            "image detail",
            "relevant location",
            "point of interest",
            "salient region",
            "examined area",
        ),
    ),
    RewriteRule(
        "fixated_fallback",
        _compile(r"\bfixated\b"),
        (
            "visually prominent",
            "visually emphasized",
            "salient",
            "prominent in the image",
            "visually notable",
        ),
    ),
    RewriteRule(
        "fixating_fallback",
        _compile(r"\bfixating\b"),
        (
            "inspecting",
            "examining",
            "looking closely at",
            "reviewing",
            "checking",
        ),
    ),
    RewriteRule(
        "coordinates_provided_by_user_artifact",
        _compile(
            r"\bcoordinates?\s+provided\s+by\s+(?:the\s+)?user\s+"
            r"(?:visual\s+cues?|image\s+details?|relevant\s+locations?|"
            r"salient\s+regions?|examined\s+areas?|pertinent\s+points?)\b"
        ),
        (
            "relevant image coordinates",
            "specific coordinates in the image",
            "pertinent visual locations",
            "useful image positions",
            "key points in the image",
        ),
    ),
    RewriteRule(
        "user_annotation_list_artifact",
        _compile(
            r"\b(?:the\s+)?user\s+(?:visual\s+cues?|image\s+details?|"
            r"relevant\s+locations?|salient\s+regions?|examined\s+areas?|"
            r"pertinent\s+points?)\s+list\s+(?:has|contains|includes)\s*:"
        ),
        (
            "Relevant image locations include:",
            "Pertinent visual areas include:",
            "Useful image coordinates include:",
            "Key points in the image include:",
            "The visible regions include:",
        ),
    ),
    RewriteRule(
        "user_annotation_artifact",
        _compile(
            r"\b(?:the\s+)?user\s+"
            r"(?P<noun>visual\s+cues?|image\s+details?|relevant\s+locations?|"
            r"salient\s+regions?|examined\s+areas?|pertinent\s+points?)\b"
        ),
        _annotation_region_replacements,
    ),
    RewriteRule(
        "missing_space_before_human_visual",
        _compile(r"(?P<prefix>\d)(?P<phrase>human\s+visual)\b"),
        lambda match, _: (f"{match.group('prefix')} {match.group('phrase')}",),
    ),
    RewriteRule(
        "human_visual_partial_attention_artifact",
        _compile(r"\b(?:the\s+)?human\s+visual\s+the\b"),
        ("the",),
    ),
    RewriteRule(
        "human_visual_annotation_artifact",
        _compile(
            r"\bhuman\s+visual\s+"
            r"(?P<noun>focus|emphasis|interest|path|cues?)\b"
        ),
        _human_visual_artifact_replacements,
    ),
    RewriteRule(
        "human_visually_emphasized",
        _compile(r"\bhuman\s+visually\s+emphasized\b"),
        (
            "visually emphasized",
            "prominent in the image",
            "visually salient",
            "notable in the image",
            "visually distinctive",
        ),
    ),
    RewriteRule(
        "generated_location_phrase_in_context",
        _compile(
            rf"\b(?P<prefix>in|looking\s+back\s+at|look\s+back\s+at|"
            rf"looking\s+at|reviewing|checking|examining|using)\s+"
            rf"(?:the\s+){{0,2}}(?P<generated>"
            rf"image\s+contains\s+several\s+relevant\s+locations|"
            rf"visible\s+content\s+contains\s+useful\s+points\s+of\s+reference|"
            rf"several\s+pertinent\s+image\s+regions\s+stand\s+out|"
            rf"several\s+relevant\s+coordinates\s+appear\s+in\s+the\s+image|"
            rf"multiple\s+useful\s+image\s+locations\s+can\s+be\s+inspected)\b"
        ),
        _generated_location_context_replacements,
    ),
    RewriteRule(
        "generated_location_phrase_before_predicate",
        _compile(
            rf"\b{GENERATED_FINITE_LOCATION_PHRASE}\s+"
            rf"(?P<predicate>includes?|including|contains?|containing|shows?|"
            rf"showing|is|are|was|were|appears?|seems?|has|have|like|"
            rf"often\s+\w+|generally\s+\w+|typically\s+\w+|usually\s+\w+)\b"
        ),
        _generated_location_predicate_replacements,
    ),
    RewriteRule(
        "duplicate_definite_article",
        _compile(r"\bthe\s+the\b"),
        ("the",),
    ),
    RewriteRule(
        "a_before_vowel_noun",
        _compile(r"\ba\s+(?P<noun>image|examined)\b"),
        lambda match, _: (f"an {match.group('noun')}",),
    ),
    RewriteRule(
        "an_before_consonant_noun",
        _compile(r"\ban\s+(?P<noun>visual|relevant|salient)\b"),
        lambda match, _: (f"a {match.group('noun')}",),
    ),
)


IMPORTANT_TO_NOTE_RE = re.compile(
    r"\bimportant\s+to\s+note\s*:\s*",
    flags=re.IGNORECASE,
)

DISPLAYED_NOTE_RE = re.compile(
    r"\b(?:(?:there\s+(?:is|is\s+also)|there['’]s|we\s+(?:see|find))\s+)"
    r"a\s+note\s*:\s*",
    flags=re.IGNORECASE,
)

NOTE_LABEL_RE = re.compile(
    r"(?:(?:\*\*|__)\s*)?\bnote\s*:(?:\s*(?:\*\*|__))?[ \t]*"
    r"(?P<lead>[\(\[\{\"'“‘*_`]*)"
    r"(?P<first>[a-z])?",
    flags=re.IGNORECASE,
)

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fixation", _compile(r"\bfixations?\b")),
    ("human_attended", _compile(r"\bhuman[- ]attended\b")),
    ("human_attention", _compile(r"\bhuman[- ]attention\b")),
    (
        "attention_artifact",
        _compile(
            r"\battention\s+(?:points?|patterns?|regions?|areas?|locations?|"
            r"coordinates?|data|maps?|cues?|clusters?|zones?|lists?|fixations?)\b"
        ),
    ),
    (
        "attended_region",
        _compile(
            r"\battended\s+(?:regions?|areas?|locations?|points?|"
            r"coordinates?|zones?)\b"
        ),
    ),
    (
        "attention_emphasis",
        _compile(
            r"(?<!my\s)(?<!our\s)(?<!your\s)\b"
            r"(?:human\s+visual\s+|visual\s+|human\s+)?attention\s+(?:was|is)\s+"
            r"(?:paid|drawn|directed|focused)\b"
        ),
    ),
    (
        "viewer_checked",
        _compile(r"\bwhere\s+the\s+(?:viewer|observer)\s+checked\s+for\b"),
    ),
    (
        "peripheral_tracking",
        _compile(r"\b(?:peripheral|visual|eye)\s+tracking\b"),
    ),
    ("fixated", _compile(r"\bfixat(?:ed|ing)\b")),
    (
        "gaze_annotation",
        _compile(
            r"\b(?:gaze\s+(?:points?|patterns?|regions?|areas?|locations?|"
            r"coordinates?|data|maps?|cues?|clusters?|zones?|lists?|fixations?)|"
            r"(?:the\s+)?(?:human|user|viewer|observer)['’]s\s+gaze"
            r"(?:\s+history)?|"
            r"(?:the\s+)?(?:human|user)\s+gaze)\b"
        ),
    ),
    ("visual_attention", _compile(r"\b(?:human\s+)?visual\s+attention\b")),
    ("eye_movement", _compile(r"\b(?:transitional\s+)?eye\s+movements?\b")),
    (
        "user_annotation_artifact",
        _compile(
            r"\b(?:the\s+)?user\s+(?:visual\s+cues?|image\s+details?|"
            r"relevant\s+locations?|salient\s+regions?|examined\s+areas?|"
            r"pertinent\s+points?)\b"
        ),
    ),
    (
        "human_visual_annotation_artifact",
        _compile(
            r"\bhuman\s+(?:visual\s+(?:the|focus|emphasis|interest|path|cues?)|"
            r"visually\s+emphasized)\b"
        ),
    ),
    (
        "generated_location_phrase_before_predicate",
        _compile(
            rf"\b{GENERATED_FINITE_LOCATION_PHRASE}\s+"
            rf"(?:includes?|including|contains?|containing|shows?|showing|is|are|"
            rf"was|were|appears?|seems?|has|have|like|often\s+\w+|"
            rf"generally\s+\w+|typically\s+\w+|usually\s+\w+)\b"
        ),
    ),
    ("duplicate_definite_article", _compile(r"\bthe\s+the\b")),
    ("note_label", _compile(r"\bnote\s*:")),
)

# This is deliberately a broad superset of every rewrite/forbidden family.  It
# avoids applying dozens of ordered expressions to ordinary responses while
# never deciding that a matching response is safe; matching text still takes
# the complete fail-closed path below.
CLEANUP_PREFILTER_RE = _compile(
    r"fixat|attent|attend|\bgaze\b|\bnote\s*:|"
    r"\beye\s+movements?\b|\btracking\b|\bviewer\b|\bobserver\b|"
    r"\buser\b|\bhuman\s+visual\b|\bhuman\s+visually\b|"
    r"\dhuman\s+visual\b|"
    rf"\bthe\s+the\b|\b{GENERATED_FINITE_LOCATION_PHRASE}\b|"
    r"\ba\s+(?:image|examined)\b|\ban\s+(?:visual|relevant|salient)\b"
)


def _stable_choice(
    options: Sequence[str],
    *,
    variation_key: str,
    rule_name: str,
    occurrence: int,
    matched_text: str,
) -> str:
    if not options:
        raise ValueError(f"rewrite rule {rule_name!r} has no replacements")
    payload = (
        f"{REPLACEMENT_VARIATION_NAMESPACE}\0{variation_key}\0{rule_name}\0"
        f"{occurrence}\0"
        f"{matched_text.casefold()}"
    ).encode("utf-8")
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(options)
    return options[index]


def _first_alpha_is_upper(text: str) -> bool:
    for char in text:
        if char.isalpha():
            return char.isupper()
    return False


def _match_initial_case(source: str, replacement: str) -> str:
    if not _first_alpha_is_upper(source):
        return replacement
    chars = list(replacement)
    for index, char in enumerate(chars):
        if char.isalpha():
            chars[index] = char.upper()
            break
    return "".join(chars)


def _remove_note_labels(
    text: str, counts: Counter[str], *, variation_key: str
) -> str:
    displayed_occurrence = 0

    def replace_important(match: re.Match[str]) -> str:
        counts["note_label_removed"] += 1
        return _match_initial_case(match.group(0), "importantly, ")

    def replace_displayed(match: re.Match[str]) -> str:
        nonlocal displayed_occurrence
        counts["note_label_removed"] += 1
        replacement = _stable_choice(
            (
                "the image states: ",
                "a caption reads: ",
                "the accompanying text says: ",
                "the image includes the text: ",
            ),
            variation_key=variation_key,
            rule_name="displayed_note",
            occurrence=displayed_occurrence,
            matched_text=match.group(0),
        )
        displayed_occurrence += 1
        return _match_initial_case(match.group(0), replacement)

    def replace(match: re.Match[str]) -> str:
        counts["note_label_removed"] += 1
        before = match.string[: match.start()]
        line_prefix = before.rsplit("\n", 1)[-1]
        stripped_prefix = re.sub(r"^[ \t>*+\-#]+", "", line_prefix).strip()
        previous = before.rstrip()[-1:] if before.rstrip() else ""
        sentence_initial = not stripped_prefix or previous in ".!?"
        first = match.group("first") or ""
        if sentence_initial:
            first = first.upper()
        else:
            first = first.lower()
        return f"{match.group('lead')}{first}"

    text = IMPORTANT_TO_NOTE_RE.sub(replace_important, text)
    text = DISPLAYED_NOTE_RE.sub(replace_displayed, text)
    return NOTE_LABEL_RE.sub(replace, text)


def remaining_language_artifacts(text: str) -> Counter[str]:
    """Return hidden-attention and Note-label artifacts still in ``text``."""

    return Counter(
        {
            name: len(pattern.findall(text))
            for name, pattern in FORBIDDEN_PATTERNS
            if pattern.search(text)
        }
    )


def clean_assistant_language(
    text: str, *, variation_key: str
) -> tuple[str, Counter[str]]:
    """Rewrite unsupported attention references in one assistant response.

    The result is deterministic for a given ``variation_key`` and input.  A
    nonempty artifact counter after the ordered rules is a programming error:
    the builder should never silently emit a partially cleaned response.
    """

    counts: Counter[str] = Counter()
    if not CLEANUP_PREFILTER_RE.search(text):
        cleaned = re.sub(r"[ \t]+([,.;!?])", r"\1", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if cleaned != text:
            counts["cleanup_format_normalization"] += 1
        return cleaned, counts

    cleaned = _remove_note_labels(text, counts, variation_key=variation_key)
    for rule in RULES:
        occurrence = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal occurrence
            choices = (
                rule.replacements(match, variation_key)
                if callable(rule.replacements)
                else rule.replacements
            )
            replacement = _stable_choice(
                choices,
                variation_key=variation_key,
                rule_name=rule.name,
                occurrence=occurrence,
                matched_text=match.group(0),
            )
            occurrence += 1
            counts[rule.name] += 1
            return _match_initial_case(match.group(0), replacement)

        cleaned = rule.pattern.sub(replace, cleaned)

    before_format_normalization = cleaned
    cleaned = re.sub(r"[ \t]+([,.;!?])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if cleaned != before_format_normalization:
        counts["cleanup_format_normalization"] += 1
    remaining = remaining_language_artifacts(cleaned)
    if remaining:
        raise ValueError(
            "assistant language cleanup left unsupported annotation references: "
            f"{dict(remaining)}"
        )
    return cleaned, counts
