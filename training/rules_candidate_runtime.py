"""Explicit opt-in adapters for isolated training/evaluation processes."""
import hashlib
from pathlib import Path

from training.rules_candidate_match import CandidateSouthMatch
from training.rules_candidate_metrics import finish_candidate_episode, summarize_candidate_match
from training.rules_candidate_parallel import ParallelGamePool
from mahjong_env.feature import FeatureAgent


RULE_SOURCES = tuple('training/rules_candidate_' + name + '.py' for name in
                     ('game','feature','match','kan','scoring','metrics','parallel','runtime'))


def rule_identity():
    root = Path(__file__).resolve().parents[1]
    return dict(rules_version=CandidateSouthMatch.rules_version,
                policy_action_contract=CandidateSouthMatch.policy_action_contract,
                observation_contract=FeatureAgent.OBSERVATION_CONTRACT,
                observation_source_sha256=hashlib.sha256((root/'mahjong_env/feature.py').read_bytes()).hexdigest(),
                rules_source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest()
                                     for p in RULE_SOURCES})


def install_training(module):
    from evaluation import battle as battle_models
    install_evaluation(battle_models)
    module.SouthMatch = CandidateSouthMatch
    module.ParallelGamePool = ParallelGamePool
    module.finish_episode = finish_candidate_episode
    install_args(module)
    validate = module.validate_resume_settings
    def validate_candidate_resume(payload, args):
        for name in ('rules_version', 'policy_action_contract',
                     'observation_contract', 'observation_source_sha256'):
            if payload.get('ppo_args', {}).get(name) != getattr(args, name):
                raise ValueError('Cannot resume a checkpoint from another rules/action/observation contract')
        validate(payload, args)
    module.validate_resume_settings = validate_candidate_resume
    for name in ('checkpoint_payload', 'policy_snapshot_payload'):
        original = getattr(module, name)
        def versioned_payload(*args, _original=original, **kwargs):
            payload = _original(*args, **kwargs)
            payload.update(rules_version=CandidateSouthMatch.rules_version,
                           policy_action_contract=CandidateSouthMatch.policy_action_contract,
                           observation_contract=FeatureAgent.OBSERVATION_CONTRACT,
                           observation_source_sha256=rule_identity()['observation_source_sha256'])
            return payload
        setattr(module, name, versioned_payload)


def install_evaluation(module):
    module.SouthMatch = CandidateSouthMatch
    module.ParallelGamePool = ParallelGamePool
    module.summarize_match = summarize_candidate_match
    install_args(module)
    signature = module.signature
    def candidate_signature(args, policies):
        return {**signature(args, policies), **rule_identity()}
    module.signature = candidate_signature


def install_args(module):
    parse = module.parse_args
    def candidate_args():
        args = parse()
        if args.game_mode != 'south':
            raise ValueError('Isolated rules entry requires --game-mode south')
        args.rules_version = CandidateSouthMatch.rules_version
        args.policy_action_contract = CandidateSouthMatch.policy_action_contract
        args.observation_contract = FeatureAgent.OBSERVATION_CONTRACT
        args.observation_source_sha256 = rule_identity()['observation_source_sha256']
        return args
    module.parse_args = candidate_args
