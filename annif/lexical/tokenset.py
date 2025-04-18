"""Index for fast matching of token sets."""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray


class TokenSet:
    """Represents a set of tokens (expressed as integer token IDs) that can
    be matched with another set of tokens. A TokenSet can optionally
    be associated with a subject from the vocabulary."""

    def __init__(
        self,
        tokens: ndarray,
        subject_id: int | None = None,
        is_pref: bool = False,
    ) -> None:
        self._tokens = frozenset(tokens)
        self.key = tokens[0] if len(tokens) else None
        self.subject_id = subject_id
        self.is_pref = is_pref

    def __len__(self) -> int:
        return len(self._tokens)

    def __iter__(self):
        return iter(self._tokens)

    @property
    def tokens(self) -> frozenset:
        return self._tokens

    def contains(self, other: TokenSet) -> bool:
        """Returns True iff the tokens in the other TokenSet are all
        included within this TokenSet."""

        return other._tokens.issubset(self._tokens)

    def __setstate__(self, state):
        # Convert _tokens to a frozenset if it's a set.
        # Might happen when using models saved using Annif 1.2 or older.
        self.__dict__ = state
        if isinstance(self._tokens, set):
            self._tokens = frozenset(self._tokens)  # pragma: no cover


class TokenSetIndex:
    """A searchable index of TokenSets (representing vocabulary terms)"""

    def __init__(self) -> None:
        self._index = collections.defaultdict(set)

    def __len__(self) -> int:
        return len(self._index)

    def add(self, tset: TokenSet) -> None:
        """Add a TokenSet into this index"""
        if tset.key is not None:
            self._index[tset.key].add(tset)

    def _find_subj_tsets(self, tset: TokenSet) -> dict[int | None, TokenSet]:
        """return a dict (subject_id : TokenSet) of matches contained in the
        given TokenSet"""

        subj_tsets = {}

        for token in tset:
            for ts in self._index[token]:
                if tset.contains(ts) and (
                    ts.subject_id not in subj_tsets
                    or not subj_tsets[ts.subject_id].is_pref
                ):
                    subj_tsets[ts.subject_id] = ts

        return subj_tsets

    def _find_subj_ambiguity(self, tsets: list[TokenSet]):
        """calculate the ambiguity values (the number of other TokenSets
        that also match the same tokens) for the given TokenSets and return
        them as a dict-like object (subject_id : ambiguity_value)"""

        # group the TokenSets by their tokens, so that TokenSets with
        # identical tokens can be considered together all in one go
        elim_tsets = collections.defaultdict(set)
        for ts in tsets:
            elim_tsets[ts.tokens].add(ts.subject_id)

        subj_ambiguity = collections.Counter()

        for tokens1, subjs1 in elim_tsets.items():
            for tokens2, subjs2 in elim_tsets.items():
                if not tokens2.issuperset(tokens1):
                    continue
                for subj in subjs1:
                    subj_ambiguity[subj] += len(subjs2) - int(subj in subjs2)

        return subj_ambiguity

    def search(self, tset: TokenSet) -> list[tuple[TokenSet, int]]:
        """Return the TokenSets that are contained in the given TokenSet.
        The matches are returned as a list of (TokenSet, ambiguity) pairs
        where ambiguity is an integer indicating the number of other TokenSets
        that also match the same tokens."""

        subj_tsets = self._find_subj_tsets(tset)
        subj_ambiguity = self._find_subj_ambiguity(subj_tsets.values())

        return [
            (ts, subj_ambiguity[subject_id]) for subject_id, ts in subj_tsets.items()
        ]
