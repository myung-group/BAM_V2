"""Old module paths of this package (before the 2026-11-16 renaming) -> new module paths.

Used when reading checkpoints: checkpoints written by the old code pickle the SOAP optimiser state as
``race_v2.training.soap_grouped.ScaleBySoapGroupedState`` (and ``race_v2.training.soap_optax...`` for the reference
implementation), so the old paths have to resolve.  The alias package ``race_v2/`` does that for plain
``pickle.load``; ``race_v2.training.utils.load_checkpoint`` maps the names itself and does not need it.
Pure data: no imports.
"""
OLD_MODULES = {
    'race_v2': 'race_v2',
    'race_v2.eval.analyze_errors': 'race_v2.eval.analyze_errors',
    'race_v2.data.atom_energies': 'race_v2.data.atom_energies',
    'race_v2.models.atomic_shells': 'race_v2.models.atomic_shells',
    'race_v2.eval.bayes_lastlayer': 'race_v2.eval.bayes_lastlayer',
    'race_v2.data.build_pkl': 'race_v2.data.build_pkl',
    'race_v2.data.build_pkl_v2': 'race_v2.data.build_pkl_v2',
    'race_v2.data.xyz_graphs': 'race_v2.data.xyz_graphs',
    'race_v2.data.loaders': 'race_v2.data.loaders',
    'race_v2.training.distributed': 'race_v2.training.distributed',
    'race_v2.eval.dump_frame_errors': 'race_v2.eval.dump_frame_errors',
    'race_v2.eval.eval_sets': 'race_v2.eval.eval_sets',
    'race_v2.export.lammps': 'race_v2.export.lammps',
    'race_v2.models.finetuning': 'race_v2.models.finetuning',
    'race_v2.data.formal_assign': 'race_v2.data.formal_assign',
    'race_v2.data.graph_fields': 'race_v2.data.graph_fields',
    'race_v2.data.graph_sizes': 'race_v2.data.graph_sizes',
    'race_v2.models.linear': 'race_v2.models.linear',
    'race_v2.training.logger': 'race_v2.training.logger',
    'race_v2.models.long_range': 'race_v2.models.long_range',
    'race_v2.eval.md_lips_analysis': 'race_v2.eval.md_lips_analysis',
    'race_v2.inference.md_run': 'race_v2.inference.md_run',
    'race_v2.data.neighbors': 'race_v2.data.neighbors',
    'race_v2.models.nequip': 'race_v2.models.nequip',
    'race_v2.data.oam_subset': 'race_v2.data.oam_subset',
    'race_v2.inference.predict': 'race_v2.inference.predict',
    'race_v2.models.race_lr': 'race_v2.models.race_lr',
    'race_v2.models.race': 'race_v2.models.race',
    'race_v2.models.race_polar': 'race_v2.models.race_polar',
    'race_v2.inference.relax_batched': 'race_v2.inference.relax_batched',
    'race_v2.inference.relax_wbm': 'race_v2.inference.relax_wbm',
    'race_v2.training.soap_grouped': 'race_v2.training.soap_grouped',
    'race_v2.eval.soap_laplace': 'race_v2.eval.soap_laplace',
    'race_v2.training.soap_optax': 'race_v2.training.soap_optax',
    'race_v2.data.stream_data': 'race_v2.data.stream_data',
    'race_v2.data.stream_io': 'race_v2.data.stream_io',
    'race_v2.data.stream_plan': 'race_v2.data.stream_plan',
    'race_v2.data.stream_worker': 'race_v2.data.stream_worker',
    'race_v2.models.thole': 'race_v2.models.thole',
    'race_v2.training.train': 'race_v2.training.train',
    'race_v2.training.utils': 'race_v2.training.utils',
    'race_v2.models.blocks': 'race_v2.models.blocks',
}
