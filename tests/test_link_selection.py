"""Link selection: which links, and what shape they come back in.

Two independent rules, and these tests keep them independent:

1. *which* - a ``Link`` names one by spacecraft, an ``int`` by position, and
   any sequence is several;
2. *what shape* - a single specifier drops the link axis, a sequence keeps it.

The type of the container carries no meaning: a bare tuple is a sequence like
a list. That is what ``Link`` bought, and losing it silently reintroduces a
footgun that survives a config-file round trip.
"""

import json

import numpy as np
import pytest

from WDM_LISAresponse.code.utils.utils import (
    Link,
    resolve_link_index,
    resolve_link_selection,
)

#: The default convention, spelled out so the expected indices below can be
#: read without cross-referencing:
#:      0 = 2->1   1 = 3->2   2 = 1->3   3 = 3->1   4 = 2->3   5 = 1->2
CONVENTION = (Link(2, 1), Link(3, 2), Link(1, 3),
              Link(3, 1), Link(2, 3), Link(1, 2))


# --------------------------------------------------------------- which links

@pytest.mark.parametrize("specifier, expected", [
    (Link(2, 1), 0),
    (Link(3, 2), 1),
    (Link(1, 2), 5),
    (0, 0),
    (3, 3),
    (-1, 5),
    (-6, 0),
])
def test_single_specifier_resolves_to_its_index(specifier, expected):
    """A Link resolves by spacecraft, an int by position.

    Negative indices count from the end, as in numpy.
    """
    assert resolve_link_index(specifier, CONVENTION) == expected


@pytest.mark.parametrize("sequence, expected", [
    ([2, 1], (2, 1)),
    ([0], (0,)),
    ([Link(2, 1)], (0,)),
    ([Link(2, 1), Link(3, 2)], (0, 1)),
    ([Link(2, 1), 3], (0, 3)),
    ([3, 0, 1], (3, 0, 1)),
])
def test_sequence_resolves_elementwise_and_keeps_order(sequence, expected):
    """Each entry resolves on its own, and the order asked for survives.

    Links and indices may be mixed freely. The result is not sorted: axis 0
    of the output follows the request.
    """
    indices, _ = resolve_link_selection(sequence, CONVENTION)
    assert indices == expected


def test_none_selects_every_link_in_convention_order():
    """None means the whole constellation, in the convention's own order."""
    indices, drop = resolve_link_selection(None, CONVENTION)

    assert indices == tuple(range(len(CONVENTION)))
    assert drop is False


# ---------------------------------------------------------------- what shape

@pytest.mark.parametrize("specifier", [Link(2, 1), 0, -1])
def test_single_specifier_drops_the_link_axis(specifier):
    """Asking for one thing returns one thing, not a stack of one."""
    _, drop = resolve_link_selection(specifier, CONVENTION)

    assert drop is True


@pytest.mark.parametrize("sequence", [
    [0], [Link(2, 1)], [2, 1], (0, 3), [Link(2, 1), 3],
])
def test_sequence_keeps_the_link_axis(sequence):
    """Asking with a sequence keeps the axis, even for a single entry.

    This is what lets code that loops over a selection work unchanged
    whether the selection holds one link or six.
    """
    _, drop = resolve_link_selection(sequence, CONVENTION)

    assert drop is False


def test_same_link_two_ways_differs_only_in_shape():
    """Link(2,1) and [Link(2,1)] pick the same arm at different rank.

    The two rules are independent: which link is chosen does not depend on
    the container, and the shape does not depend on the specifier's kind.
    """
    scalar_indices, scalar_drop = resolve_link_selection(
        Link(2, 1), CONVENTION)
    boxed_indices, boxed_drop = resolve_link_selection(
        [Link(2, 1)], CONVENTION)

    assert scalar_indices == boxed_indices == (0,)
    assert scalar_drop is True
    assert boxed_drop is False


# ------------------------------------------- the container type means nothing

@pytest.mark.parametrize("as_tuple, as_list", [
    ((0, 3), [0, 3]),
    ((4, 5), [4, 5]),
    ((1, 2, 3), [1, 2, 3]),
    ((0,), [0]),
])
def test_a_bare_tuple_is_a_sequence_like_a_list(as_tuple, as_list):
    """Tuple and list are interchangeable containers.

    Losing this reintroduces tuple-vs-list dispatch, which changes meaning
    silently whenever a value passes through JSON or a config file.
    """
    assert (resolve_link_selection(as_tuple, CONVENTION)
            == resolve_link_selection(as_list, CONVENTION))


@pytest.mark.parametrize("pair", list(CONVENTION))
def test_a_bare_pair_that_is_also_a_link_is_refused(pair):
    """The one genuinely ambiguous input, refused rather than guessed at.

    A two-integer tuple that is also a link in the convention reads equally
    as that link and as two indices, so neither reading is imposed.
    """
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_link_selection(tuple(pair), CONVENTION)


def test_the_refusal_names_both_ways_out():
    """The error tells the caller how to say either thing they meant."""
    with pytest.raises(ValueError, match=r"Link\(2, 1\).*\[2, 1\]"):
        resolve_link_selection((2, 1), CONVENTION)


def test_json_round_trip_is_unambiguous():
    """A link written to a config file comes back as indices, plainly.

    JSON has no tuples, so a pair becomes a list. Under the old dispatch
    that silently changed which links were computed.
    """
    round_tripped = json.loads(json.dumps(list(Link(2, 1))))

    assert round_tripped == [2, 1]
    assert resolve_link_selection(round_tripped, CONVENTION) == ((2, 1), False)


# ------------------------------------------------------------------ the type

def test_link_is_tuple_compatible():
    """A convention built from plain pairs still resolves.

    Link is a NamedTuple, so it compares equal to the pair it names and
    callers are not forced to adopt the type.
    """
    plain = tuple(tuple(link) for link in CONVENTION)

    assert resolve_link_index(Link(2, 1), plain) == 0
    assert resolve_link_selection(None, plain)[0] == tuple(range(6))


def test_link_unpacks():
    """Link unpacks like the pair it is."""
    sender, receiver = Link(2, 1)

    assert (sender, receiver) == (2, 1)


def test_link_repr_shows_the_direction():
    """The repr reads as a direction, so tracebacks are legible."""
    assert repr(Link(2, 1)) == "Link(2->1)"


# ------------------------------------------------------------------- refusal

@pytest.mark.parametrize("index", [6, -7, 99])
def test_out_of_range_index_raises(index):
    """An index outside the convention is rejected, with the range shown."""
    with pytest.raises(AssertionError, match="Link index must lie in"):
        resolve_link_index(index, CONVENTION)


@pytest.mark.parametrize("absent", [Link(1, 1), Link(2, 2), Link(9, 8)])
def test_a_link_outside_the_convention_raises_and_shows_it(absent):
    """A Link the convention does not contain is rejected, not silently
    matched to something near it."""
    with pytest.raises(AssertionError, match="not in the link convention"):
        resolve_link_index(absent, CONVENTION)


@pytest.mark.parametrize("bad", ["21", 2.5, None, {"sender": 2}])
def test_a_wrong_type_raises_and_names_link(bad):
    """A specifier of the wrong type is refused, and the message says what
    to write instead."""
    with pytest.raises(TypeError, match="Link"):
        resolve_link_index(bad, CONVENTION)


def test_an_empty_selection_raises():
    """Selecting nothing is a mistake, not an empty result."""
    with pytest.raises(AssertionError, match="At least one link"):
        resolve_link_selection([], CONVENTION)


def test_numpy_integers_are_accepted_as_indices():
    """A numpy integer works wherever a Python int does.

    Indices routinely arrive from numpy, and rejecting them would be a
    surprise with no benefit.
    """
    assert resolve_link_index(np.int64(3), CONVENTION) == 3
    assert resolve_link_selection(np.int64(3), CONVENTION) == ((3,), True)
