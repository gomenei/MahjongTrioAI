"""Explicit multi-winner metrics for the isolated full-match candidate."""
from collections import defaultdict

from training.rules_candidate_scoring import hand_winners


def seat_outcomes(game, seat):
    wins = [h for h in game.history if seat in hand_winners(h)]
    return dict(win=len(wins),
                tsumo_win=sum(h.win_by == '自摸' for h in wins),
                ron_win=sum(h.win_by != '自摸' for h in wins),
                deal_in=sum(h.discarder == seat for h in game.history),
                draw=sum(not hand_winners(h) for h in game.history),
                invalid=sum('非法动作' in h.message for h in game.history),
                winning_points=sum(max(0, h.deltas[seat]) for h in wins))


def summarize_candidate_match(item, game):
    from battle_models import empty_model_result
    by_model = defaultdict(empty_model_result)
    for seat, model_index in enumerate(item.task.seat_models):
        values = by_model[model_index]
        rank = int(game.final_ranks[seat])
        outcome = seat_outcomes(game, seat)
        values['hands'] += 1
        values['played_hands'] += game.hand_count
        values['rank_sum'] += rank
        values[('firsts', 'seconds', 'thirds')[rank-1]] += 1
        values['utility_sum'] += {1: 1., 2: 0., 3: -1.}[rank]
        values['point_sum'] += game.scores[seat] - game.starting_scores[seat]
        for key, source in (('wins','win'), ('tsumo_wins','tsumo_win'),
                            ('ron_wins','ron_win'), ('deal_ins','deal_in'),
                            ('draws','draw'), ('invalid_actions','invalid'),
                            ('winning_points_sum','winning_points')):
            values[key] += outcome[source]
    return by_model


def finish_candidate_episode(episode, args):
    game = episode.game
    if not game.done or game.final_ranks is None:
        raise ValueError('Candidate episode requires a completed full match')
    seat = episode.learner_seat
    points = game.scores[seat] - game.starting_scores[seat]
    rank = game.final_ranks[seat]
    raw = points / args.reward_scale + args.rank_reward_weight * {1:1., 2:0., 3:-1.}[rank]
    outcome = seat_outcomes(game, seat)
    return dict(episode=episode, points=points,
                reward=max(-args.reward_clip, min(args.reward_clip, raw)),
                win=outcome['win'], ron_win=outcome['ron_win'], tsumo_win=outcome['tsumo_win'],
                first=int(rank == 1), rank=rank, deal_in=outcome['deal_in'],
                draw=outcome['draw'], invalid=int(outcome['invalid'] > 0),
                played_hands=game.hand_count)
