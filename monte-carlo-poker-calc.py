"""Monte Carlo Texas Hold'em equity calculator.

Usage:
    python equity.py AhAs KdKc
    python equity.py AhKh QsQd 7c8c --board Qh9h2c --trials 100000
"""
import argparse
import math
import random
from collections import Counter

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [(r, s) for r in range(2, 15) for s in SUITS]  # rank 2..14 (A=14)


def parse_card(text):
    """'Ah' -> (14, 'h')."""
    rank, suit = text[0].upper(), text[1].lower()
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"bad card: {text}")
    return (RANKS.index(rank) + 2, suit)


def parse_cards(text):
    """'AhKd' -> [(14,'h'), (13,'d')]."""
    if len(text) % 2:
        raise ValueError(f"bad card string: {text}")
    return [parse_card(text[i:i + 2]) for i in range(0, len(text), 2)]


def straight_high(ranks):
    """Top card of the best straight in `ranks`, or 0 if none. Ace also plays low."""
    rs = set(ranks)
    if 14 in rs:
        rs.add(1)
    for high in range(14, 4, -1):
        if all(high - i in rs for i in range(5)):
            return high
    return 0


def evaluate(cards):
    """Score the best 5-card hand from 7 cards.

    Returns a tuple (category, tiebreakers...). Python compares tuples
    lexicographically, so a higher tuple is a stronger hand.
    Categories: 8 straight flush, 7 quads, 6 full house, 5 flush,
    4 straight, 3 trips, 2 two pair, 1 pair, 0 high card.
    """
    ranks = [r for r, _ in cards]
    counts = Counter(ranks)
    suit_counts = Counter(s for _, s in cards)
    flush_suit = next((s for s, n in suit_counts.items() if n >= 5), None)

    if flush_suit:
        flush_ranks = sorted((r for r, s in cards if s == flush_suit), reverse=True)
        sf = straight_high(flush_ranks)
        if sf:
            return (8, sf)

    # (rank, count) sorted by count then rank, both descending
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    top_rank, top_count = groups[0]
    second_rank, second_count = groups[1]

    if top_count == 4:
        kicker = max(r for r in ranks if r != top_rank)
        return (7, top_rank, kicker)
    if top_count == 3 and second_count >= 2:
        return (6, top_rank, second_rank)
    if flush_suit:
        return (5, *flush_ranks[:5])
    st = straight_high(ranks)
    if st:
        return (4, st)
    if top_count == 3:
        kickers = sorted((r for r in counts if r != top_rank), reverse=True)[:2]
        return (3, top_rank, *kickers)
    if top_count == 2 and second_count == 2:
        kicker = max(r for r in counts if r not in (top_rank, second_rank))
        return (2, top_rank, second_rank, kicker)
    if top_count == 2:
        kickers = sorted((r for r in counts if r != top_rank), reverse=True)[:3]
        return (1, top_rank, *kickers)
    return (0, *sorted(ranks, reverse=True)[:5])


def equity(hands, board=(), trials=50_000, seed=None):
    """Estimate each hand's equity by simulating random runouts of the board.

    hands: list of 2-card lists, board: 0-5 known cards.
    Ties split the pot: each of k tied winners gets 1/k of that trial.
    Returns (equities, standard_errors).
    """
    used = [c for h in hands for c in h] + list(board)
    if len(set(used)) != len(used):
        raise ValueError("duplicate card")
    deck = [c for c in FULL_DECK if c not in set(used)]
    to_deal = 5 - len(board)
    rng = random.Random(seed)

    shares = [0.0] * len(hands)
    for _ in range(trials):
        full_board = list(board) + rng.sample(deck, to_deal)
        scores = [evaluate(h + full_board) for h in hands]
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        for i in winners:
            shares[i] += 1 / len(winners)

    eq = [s / trials for s in shares]
    # Standard error of a mean of values in [0,1]: <= sqrt(p(1-p)/N)
    se = [math.sqrt(p * (1 - p) / trials) for p in eq]
    return eq, se


def main():
    ap = argparse.ArgumentParser(description="Monte Carlo hold'em equity")
    ap.add_argument("hands", nargs="+", help="hole cards, e.g. AhKd")
    ap.add_argument("--board", default="", help="known board cards, e.g. Qh9h2c")
    ap.add_argument("--trials", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    hands = [parse_cards(h) for h in args.hands]
    if any(len(h) != 2 for h in hands):
        raise SystemExit("each hand needs exactly 2 cards")
    board = parse_cards(args.board)
    if len(board) > 5:
        raise SystemExit("board has at most 5 cards")

    eq, se = equity(hands, board, args.trials, args.seed)
    for text, p, e in zip(args.hands, eq, se):
        print(f"{text:6} {100 * p:6.2f}%  ± {196 * e:.2f}% (95% CI)")


if __name__ == "__main__":
    main()