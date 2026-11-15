"""Old module paths of this package (before the 2026-11-16 renaming) -> new module paths.

Used when reading checkpoints: checkpoints written by the old code pickle the SOAP optimiser state as
``bam_omat24.soap_grouped.ScaleBySoapGroupedState`` (and ``bam_omat24.soap_optax...`` for the reference
implementation), so the old paths have to resolve.  The alias package ``bam_omat24/`` does that for plain
``pickle.load``; ``race_v2.training.utils.load_checkpoint`` maps the names itself and does not need it.
Pure data: no imports.
"""
OLD_MODULES = {
    'bam_omat24': 'race_v2',
    'bam_omat24.analyze_errors': 'race_v2.eval.analyze_errors',
    'bam_omat24.atom_energies': 'race_v2.data.atom_energies',
    'bam_omat24.atomic_shells': 'race_v2.models.atomic_shells',
    'bam_omat24.bayes_lastlayer': 'race_v2.eval.bayes_lastlayer',
    'bam_omat24.build_pkl': 'race_v2.data.build_pkl',
    'bam_omat24.build_pkl_v2': 'race_v2.data.build_pkl_v2',
    'bam_omat24.data_3bpa': 'race_v2.data.xyz_graphs',
    'bam_omat24.data_nnx': 'race_v2.data.loaders',
    'bam_omat24.distributed': 'race_v2.training.distributed',
    'bam_omat24.dump_frame_errors': 'race_v2.eval.dump_frame_errors',
    'bam_omat24.eval_sets': 'race_v2.eval.eval_sets',
    'bam_omat24.export_lammps': 'race_v2.export.lammps',
    'bam_omat24.finetuning_nnx': 'race_v2.models.finetuning',
    'bam_omat24.formal_assign': 'race_v2.data.formal_assign',
    'bam_omat24.graph_fields': 'race_v2.data.graph_fields',
    'bam_omat24.graph_sizes': 'race_v2.data.graph_sizes',
    'bam_omat24.linear_nnx': 'race_v2.models.linear',
    'bam_omat24.logger': 'race_v2.training.logger',
    'bam_omat24.long_range': 'race_v2.models.long_range',
    'bam_omat24.md_lips_analysis': 'race_v2.eval.md_lips_analysis',
    'bam_omat24.md_run': 'race_v2.inference.md_run',
    'bam_omat24.neighbors': 'race_v2.data.neighbors',
    'bam_omat24.nequip_nnx': 'race_v2.models.nequip',
    'bam_omat24.oam_subset': 'race_v2.data.oam_subset',
    'bam_omat24.predict': 'race_v2.inference.predict',
    'bam_omat24.race_lr_nnx': 'race_v2.models.race_lr',
    'bam_omat24.race_nnx': 'race_v2.models.race',
    'bam_omat24.race_polar_nnx': 'race_v2.models.race_polar',
    'bam_omat24.relax_batched': 'race_v2.inference.relax_batched',
    'bam_omat24.relax_wbm': 'race_v2.inference.relax_wbm',
    'bam_omat24.soap_grouped': 'race_v2.training.soap_grouped',
    'bam_omat24.soap_laplace': 'race_v2.eval.soap_laplace',
    'bam_omat24.soap_optax': 'race_v2.training.soap_optax',
    'bam_omat24.stream_data': 'race_v2.data.stream_data',
    'bam_omat24.stream_io': 'race_v2.data.stream_io',
    'bam_omat24.stream_plan': 'race_v2.data.stream_plan',
    'bam_omat24.stream_worker': 'race_v2.data.stream_worker',
    'bam_omat24.thole': 'race_v2.models.thole',
    'bam_omat24.train_3bpa': 'race_v2.training.train',
    'bam_omat24.training_utils': 'race_v2.training.utils',
    'bam_omat24.utils': 'race_v2.models.blocks',
}
