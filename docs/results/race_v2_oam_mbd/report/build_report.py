"""RACE v2: methods (with equations) and Matbench Discovery results (MPtrj-only and OAM) -- PDF.

usage: python -I build_report.py <font_dir> <fig_dir> <out.pdf> [final_results.json]
Equations follow the code the models were trained with (branch racev2, old package name bam_omat24); see methods_formulas.md.
final_results.json (optional): numbers of the final OAM checkpoint, see FINAL below; without it the latest measured
checkpoint (virtual epoch 22, float64 evaluation) is shown and marked as interim.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdfkit import Kit                                                                  # noqa: E402

FONT_DIR, FIG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
# OAM row of the tables: label, virtual epoch, F1, MAE, R2, RMSD, kappa SRME, CPS, validation E/F MAE, 3k points
FINAL = dict(label='virtual epoch 22 (interim model, before LR decay)', interim=True, epoch=22, f1=0.887, mae=0.0220, r2=0.869,
             rmsd=0.0644, ksrme=0.220, cps=0.857, prec=0.880, rank_cps='20', val='11.3 / 36.6', hours='15.9')
if len(sys.argv) > 4:
    FINAL.update(json.load(open(sys.argv[4])))
B = lambda s: f'<font name="NanumB">{s}</font>'                                          # noqa: E731
K = Kit(FONT_DIR, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets'))
p, h1, h2, h3, eq, bl, tb, fig, note = K.p, K.h1, K.h2, K.h3, K.eq, K.bullets, K.table, K.figure, K.note
oam = [f"{FINAL['f1']:.3f}", f"{FINAL['mae']:.4f}", f"{FINAL['r2']:.3f}", f"{FINAL['rmsd']:.4f}", f"{FINAL['ksrme']:.3f}",
       f"{FINAL['cps']:.3f}"]

# =====================================================================================================================
K.title('RACE v2: Methods and Matbench Discovery Results',
        '2026-11-16 · model RACE (17,757,551 parameters) · training code BAM_V2 (branch racev2, old package name bam_omat24) · '
        'KISTI HANGANG GH200, 2026-11-11~16')

h1('Summary')
p('RACE v2 is an equivariant message-passing interatomic potential. In this compute window we built two versions of a 7 Å cutoff '
  'model, one trained on MPtrj only (MPtrj-only) and one pretrained on OMat24 and then fine-tuned on MPtrj and sAlex (OAM), '
  'and evaluated them the Matbench Discovery way. This document gives the implemented methods, with the equations exactly as '
  'in the code, and reports the experiments and results.')
tb([['Model', 'F1', 'MAE', 'R<super>2</super>', 'RMSD', 'κ SRME', 'CPS'],
    [B('RACE v2 OAM') + f"<br/>{FINAL['label']}"] + [B(x) for x in oam],
    [B('RACE v2 MPtrj-only')] + [B(x) for x in ['0.777', '0.040', '0.800', '0.0785', '0.375', '0.761']],
    [B('BAM-MP-DeNS') + ' (lab V1, MPtrj)'] + [B(x) for x in ['0.761', '0.043', '0.797', '0.0832', '0.766', '0.672']],
    [B('BAM-OAM') + ' (lab V1, OAM)', '-', B('0.027'), '-', '-', '-', '-']],
   widths=[0.40, 0.09, 0.10, 0.09, 0.10, 0.12, 0.10],
   caption='Key results on the 215,488 unique prototypes of the full WBM set; MAE in eV/atom, RMSD in Å. '
           'CPS = 0.5 F1 + 0.4 (1 − κ SRME/2) + '
           '0.1 (1 − RMSD/0.15). The BAM values are from the leaderboard and lab records.', align_right_from=1)
bl([B('MPtrj-only:') + ' With the same data and the same evaluation method, CPS is 0.089 higher than the lab V1 (BAM-MP-DeNS). '
    'Most of the difference comes from the thermal conductivity (κ). The difference in the energy metrics (F1, MAE) is small. '
    'It ranks 6th among the 19 leaderboard models that use MPtrj only.',
    B('OAM:') + f" 2 epochs of OMat24 pretraining and MPtrj+sAlex fine-tuning give MAE {FINAL['mae']:.4f} eV/atom and CPS {FINAL['cps']:.3f}. "
    f"MAE is below BAM-OAM (0.027) and CPS is about rank {FINAL['rank_cps']} of all 49 leaderboard models.",
    B('Found along the way:') + ' (1) Fixing a fine-tuning problem in which the 8 copies of MPtrj were read clustered in one place '
    'lowered the WBM MAE from 0.0267 to 0.0225. (2) For a model trained in float32, κ is 0.861 when computed in float32 and 0.220 '
    'in float64. Phonon calculations need float64.',
    B('Still lacking:') + ' There is a gap to the top leaderboard models (CPS 0.90 or higher, MAE 0.017~0.018). The pretraining is '
    'short at 2 epochs and the model is on the small side at 17.8 million parameters.'])

# =====================================================================================================================
h1('1. Model overview')
p('The total energy is the sum of per-atom energies. The per-atom energy is a short-range pair repulsion term plus the three scalar '
  'energies output by each of the 5 layers, scaled and shifted by the learned scalars s and b. Forces and stress are obtained by '
  'one automatic differentiation of the total energy with respect to coordinates and lattice, so they are exactly consistent with '
  'the energy. The model has no long-range terms such as charge, polarisation or dispersion (these are covered in the '
  'RACE v2 LR-POLAR document).')
eq(r'E=\sum_i\left[s\left(e^{\mathrm{rep}}_i+\sum_{t=0}^{4}\left(e^{\mathrm{node}}_{t,i}+e^{x}_{t,i}+e^{f}_{t,i}\right)\right)+b\right]'
   r'+\sum_i E^{0}_{Z_i}')
eq(r'\mathbf{F}_i=-\dfrac{\partial E}{\partial \mathbf{x}_i},\qquad '
   r'\sigma_{uv}=\dfrac{1}{V}\left[\sum_a\dfrac{\partial E}{\partial h_{au}}h_{av}+\sum_i\dfrac{\partial E}{\partial x_{iu}}x_{iv}\right]')
tb([['Item', 'Value'],
    ['Architecture', '5 layers, 128 channels, spherical tensors l = 0~3 (2,048-dimensional features per atom)'],
    ['Cutoff', '7 Å (mean number of neighbours: MPtrj 92, OMat24 63)'],
    ['Distance input', '12 Bessel functions + 16 atomic-shell exponential functions, polynomial cutoff p = 6'],
    ['Elements', '89 (Z = 1~83, 89~94). Per-element weight tables are factorised with rank 32'],
    ['Parameters', '17,757,551. Of these, 72 % are in per-element tables (12.72 million after factorisation)'],
    ['Output', 'Energy, forces, stress (periodic boundaries)']],
   widths=[0.2, 0.8], caption='Model settings. The MPtrj-only and OAM models use the same architecture.', align_right_from=-1)
p('h is the lattice matrix (rows are lattice vectors) and V the volume. Units are Å and eV, with stress in eV/Å<super>3</super>. '
  'E<super>0</super> is the fixed per-element reference energy, subtracted from the labels in training and added '
  'back in inference.')

# =====================================================================================================================
h1('2. Methods')
h2('2.1 Input: graph and distance/angle features')
p('A graph is built with atoms as nodes and atom pairs within 7 Å as edges. With periodic boundaries the relative position includes '
  'the lattice shift s. The distance input is 28 numbers. The 12 Bessel functions are multiplied by fixed Gaussian coefficients so '
  'that the high-frequency components start small, and 16 exponential functions built from atomic shell radii are added. All are '
  'multiplied by the polynomial cutoff function and go smoothly to 0 at 7 Å.')
eq([r'\mathbf{r}_{ij}=\mathbf{x}_j-\mathbf{x}_i-\mathbf{s}_{ij}\mathbf{h},\qquad '
    r'B_n(r)=\dfrac{2}{r_c}\,\dfrac{\sin(n\pi r/r_c)}{r}\quad(n=1,\ldots,12),\qquad r_c=7\ \mathrm{\AA}',
    r'f_{\mathrm{cut}}(r)=1-28u^{6}+48u^{7}-21u^{8},\qquad u=\dfrac{r}{r_c}'])
eq([r'\beta_n(r)=B_n(r)\,f_{\mathrm{cut}}(r)\,e^{-(0.224\,n)^{2}},\qquad '
    r'\varsigma_{ab}(r)=\exp\left(-\dfrac{r}{\rho_a(Z_i)+\rho_b(Z_j)}\right)f_{\mathrm{cut}}(r),\quad a,b\in\{s,p,d,f\}'])
bl(['ρ<sub>a</sub>(Z) is the mean radius of the outermost a shell of element Z (table from LDA atomic calculations, fixed values). '
    'If the element has no such shell, that channel is 0.',
    'The angular input is the real spherical harmonics Y of the unit vector (l = 0~3, 16 components).',
    'The element embedding e is an 89 × 128 learned matrix that multiplies the one-hot vector.'])
p('At very short distances, where training data are scarce, the repulsion comes from a pair potential built from UFF table values. '
  'A sigmoid switches it off beyond 1.25 times the sum of the covalent radii.')
eq(r'e^{\mathrm{rep}}_i=\dfrac{1}{2}\sum_{j\in N(i)}D_{ij}\,\dfrac{6}{\zeta_{ij}-6}\exp\left[\zeta_{ij}\left(1-\dfrac{r_{ij}}{x_{ij}}\right)\right]'
   r'f_{\mathrm{cut}}(r_{ij})\;\mathrm{sigmoid}\left(\dfrac{r_{ij}-1.25\,(R_i+R_j)}{0.3\ \mathrm{\AA}}\right)')

h2('2.2 Message-passing layer')
p('Layer t updates the features with the message m, which gathers the neighbour features and directions through a tensor product, '
  'with x, which is built from the atomic basis A, and with p, the tensor product of the two. Neighbour sums are divided by the mean '
  'number of neighbours. The atomic basis A is the layer-0 message built from the element embeddings alone and is computed only once.')
eq([r'A_i=\dfrac{1}{\bar{n}}\sum_{j\in N(i)}R^{(0)}(r_{ij})\odot\left[W^{(0)}_1 e_j\otimes Y(\hat{\mathbf{r}}_{ij})\right],\qquad '
    r'm_i=\dfrac{1}{\bar{n}}\sum_{j\in N(i)}R^{(t)}(r_{ij})\odot\left[W_1 h_j\otimes Y(\hat{\mathbf{r}}_{ij})\right]'])
eq([r'x_i=W_x A_i,\qquad q_i=\mathrm{mean}\left(x_i^{2}\right)+10^{-6},\qquad '
    r'\hat{x}_i=\dfrac{x_i}{\sqrt{q_i}}\;2\,\mathrm{sigmoid}\left(\dfrac{1}{2}a_{t,Z_i}\ln q_i+b_{t,Z_i}\right)'])
eq([r'p_i=W_2\left[(W_m m_i)\otimes\hat{x}_i\right],\qquad \delta_i=\gamma\odot\left(p_i+S_{Z_i}h_i\right)',
    r'h^{(t+1)}=\delta\quad(t=0,1),\qquad h^{(t+1)}=2\delta-h^{(t-1)}\quad(t\geq 2)'])
bl(['R<super>(t)</super> is a distance MLP learned separately for each layer (28 → 64 → 64 → 64 → number of tensor-product paths × 128, SiLU).',
    B('Normalisation of x:') + ' x is divided by its root mean square for each atom and then multiplied by a bounded gate learned per '
    'element and per layer (initial value 1). This keeps the magnitude of x from varying between environments with many and with '
    'few neighbours.',
    B('Chebyshev residual:') + ' From the third layer on, a recurrence of the form h<super>(t+1)</super> = 2δ − h<super>(t−1)</super> '
    'is used. γ is a learned per-channel coefficient (initial value 0.5).',
    'The feature update itself contains no non-linear function. The non-linearity comes from the gate on x, the distance MLP and the '
    'energy readout.',
    'The gather, tensor product and add of the message run in a single fused cuEquivariance kernel.'])

h2('2.3 Energy readout')
p('Three scalar energies are read from each layer and summed: the node term read from the layer output, the x term read from the '
  'normalised x, and the f term read from the layer input. '
  'LN is layer normalisation, and the first linear layer of the x term and of the f term uses per-element weights.')
eq(r'e^{\mathrm{node}}_{t,i}=w\cdot\mathrm{silu}\left(W_r h^{(t+1)}_i+b\right),\qquad '
   r'e^{x}_{t,i}=\mathrm{MLP}^{x}_{Z_i}\left(\dfrac{\mathrm{LN}(\hat{x}_i)}{\sqrt{\bar{n}}}\right),\qquad '
   r'e^{f}_{t,i}=\mathrm{MLP}^{f}_{Z_i}\left(\dfrac{\mathrm{LN}(h^{(t)}_i)}{\sqrt{\bar{n}}}\right)')

h2('2.4 Rank factorisation of the per-element tables')
p('Keeping a separate weight matrix W[Z] for each element gives several tables of size 89 elements × 128 × 128, '
  'which hold most of the parameters. These were replaced by linear combinations of K = 32 shared bases.')
eq(r'W[Z]=\dfrac{1}{\sqrt{K}}\sum_{k=1}^{K}c[Z,k]\,B[k],\qquad K=32')
bl(['Factorised tables: the skip matrix S[Z] of each layer and the first linear layer of the x and f readouts.',
    'One table shrinks from 1,458,176 to 527,136 parameters, and the total from 45.9 million to 17.8 million.',
    'Tables that do not reach the energy (such as the l > 0 skip of the last layer) were reduced to rank 1.',
    'In a like-for-like comparison at a 5 Å cutoff, the validation error of the factorised model was 2~7 % lower than that of the '
    'original model (Section 3.1).'])

h2('2.5 Loss')
p('A Huber function is applied to the residuals of energy (per atom), forces (per component) and stress (per Voigt component). '
  'The weights are 1 : 1 : 1.')
eq([r'\mathcal{L}=\mathcal{L}_E+\mathcal{L}_F+\mathcal{L}_S,\qquad '
    r'H_{\delta}(x)=\dfrac{x^{2}}{2}\ \ (|x|\leq\delta),\qquad H_{\delta}(x)=\delta\left(|x|-\dfrac{\delta}{2}\right)\ \ (|x|>\delta)',
    r'\mathcal{L}_E=\dfrac{1}{G}\sum_g H_{0.05}\left(\dfrac{E^{\mathrm{pred}}_g-E^{\mathrm{ref}}_g}{N_g}\right),\qquad '
    r'\mathcal{L}_F=\dfrac{1}{3N}\sum_i\sum_{c}H_{0.25}\left(\Delta F_{i,c}\right),\qquad '
    r'\mathcal{L}_S=\dfrac{1}{G}\sum_g\sum_{v=1}^{6}H_{0.05}\left(\Delta\sigma_{g,v}\right)'])
bl(['The Huber thresholds are 0.05 eV/atom for energy, 0.25 eV/Å for forces and 0.05 eV/Å<super>3</super> for stress. These values '
    'and the weights are taken unchanged from the BAM-MP-DeNS settings.',
    'Before they enter the Huber function, the residuals are squashed once more with tanh (energy 5 eV/atom, forces 10 eV/Å, stress '
    '0.5 eV/Å<super>3</super>) so that extreme structures cannot destabilise the training.',
    'The validation loss is the same expression computed with the exponential moving average (EMA) weights. Evaluation uses the EMA '
    'weights of the last epoch.'])
note('Note: the stress term is a sum over the 6 Voigt components, so it enters 6 times larger than it would if defined as the '
     'component mean. The loss column of the training log does not include the stress term.')

h2('2.6 Optimiser and learning rate')
p('The optimiser is SOAP. It applies Adam in the eigenbasis of the Kronecker factors of the gradient and updates the basis every '
  '10 steps. The per-element coefficient tables and the 0- and 1-dimensional parameters are updated with Adam. The gradient norm is '
  'clipped at 1.0 and the EMA decay is 0.999.')
p('The learning rate is held constant after the warm-up and then reduced linearly in the final phase. The step s<sub>0</sub> at which '
  'the learning-rate decay starts and the step T at which it ends can be set during training, which makes it easy to fit the length '
  'to the available compute time.')
eq(r'f(t)=f_{\mathrm{base}}(t)\ \ (t<s_0),\qquad '
   r'f(t)=f_{\mathrm{base}}(s_0)+\left(\mu-f_{\mathrm{base}}(s_0)\right)\dfrac{t-s_0}{T-s_0}\ \ (s_0\leq t\leq T),\qquad \mu=0.01')
tb([['Training', 'Peak learning rate', 'Schedule', 'Precision'],
    ['MPtrj-only', '0.01', 'Warm-up 2 epochs, cosine (80-epoch basis) for 20 epochs, then linear decay over 20 epochs', 'All float64'],
    ['OMat24 pretraining', '0.01', 'Warm-up 10,000 steps, constant, linear decay over the last 31 %', 'Model float32, SOAP state float64'],
    ['OAM fine-tuning', '0.005', 'Warm-up 5,000 steps, constant, linear decay over the last 29 %', 'Model float32, SOAP state float64']],
   widths=[0.17, 0.13, 0.46, 0.24], caption='Learning-rate schedule and precision of each training run.', align_right_from=-1)
bl(['f<sub>base</sub> is the factor of the warm-up and cosine (or constant) phase. MPtrj-only started with cosine and was switched to '
    'linear decay at epoch 20.',
    'A float32 model with only the SOAP statistics in float64 is about 1.8 times faster than all float64, with the same validation error.'])

h2('2.7 Data reading and shuffling')
p('Data are read directly from LMDB on disk (streaming). GPU batches are filled using a size table that holds the precounted numbers of '
  'atoms and edges of every structure. The 7 Å size table of the OMat24 training set (100 million structures) was built '
  'in five parts on five nodes (about 30 minutes per node).')
p('The fine-tuning data are 8 copies of the MPtrj training set combined with 1 copy of the sAlex training set. This is the same '
  'composition that the OMat24 paper used for the OAM models of eSEN and EquiformerV2. One epoch is about 22.43 million structures, '
  '54 % MPtrj and 46 % sAlex.')
bl([B('Problem fixed:') + ' In the first implementation the 8 repeated copies went into the same read block, so the 8 copies of one '
    'structure came out clustered within about 21 steps. In effect the MPtrj part of a batch was filled with a small number of '
    'structures. The fix gives each copy a different file number so that the copies are spread over the whole epoch '
    '(median span over which the 8 copies appear: 80 % of the epoch).',
    'The fix moved full-WBM MAE from 0.0267 to 0.0225 eV/atom, F1 from 0.844 to 0.886 (Section 3.3).'])

K.need(230)
h2('2.8 Inference precision')
p('A model trained in float32 can also be run with its parameters promoted to float64. We measured which calculations need this.')
tb([['Calculation', 'float32', 'float64', 'Verdict'],
    ['κ SRME (103 phonon materials)', '0.861', '0.220', 'float64 required'],
    ['· materials whose relaxation hit the 300-step limit', '33', '0', ''],
    ['· imaginary vibrational modes', '6', '0', ''],
    ['WBM 3k sample MAE (eV/atom)', '0.0199', '0.0195', '2 % difference'],
    ['WBM 3k sample F1', '0.893', '0.893', 'no difference'],
    ['WBM 3k sample RMSD (Å)', '0.0647', '0.0631', '2.5 % difference']],
   widths=[0.42, 0.17, 0.17, 0.24], caption='The same checkpoint (fine-tuning virtual epoch 18) computed in float32 and in float64.',
   align_right_from=1)
bl(['In float32 the force noise is about 10<super>−4</super> eV/Å, the same size as the convergence criterion of the κ calculation '
    '(10<super>−4</super> eV/Å) and as the force differences of the finite displacements. The relaxation therefore does not converge '
    'and κ comes out smaller than it really is (a factor of 1.7 in the median).',
    'The final OAM numbers in this document were computed in float64 for both WBM and κ. The MPtrj-only model was trained and '
    'evaluated in float64.'])

h2('2.9 Evaluation method')
bl([B('WBM stability:') + ' The 256,963 initial WBM structures are relaxed with the model (FIRE, lattice and atomic positions, up to '
    '500 steps until forces reach 0.05 eV/Å), and the MP2020 correction is applied to obtain the energy above the convex hull. The '
    'metrics are computed on the 215,488 unique prototypes. No correction of any kind was applied to the model output.',
    B('RMSD:') + ' The distance between the relaxed structure and the DFT-relaxed structure (pymatgen StructureMatcher). Structures '
    'that could not be matched count as 1.0.',
    B('κ SRME:') + ' Lattice thermal conductivity of the 103 PhononDB materials. Displacement 0.03 Å, relaxation convergence '
    '10<super>−4</super> eV/Å, symmetry tolerance '
    '10<super>−5</super>. For large supercells only the batch size computed at once was reduced to fit the GPU memory.',
    B('3k tracking sample:') + ' To follow the progress during training we used 2,976 structures (2,536 unique prototypes) drawn from '
    'WBM by a fixed rule. Its MAE tends to come out about 0.002 eV/atom lower than the full result, so all final numbers were '
    'computed on the full WBM set.',
    B('Fairness:') + ' Training used only the training split of each dataset. The data ratio and training length were not '
    'chosen by looking at WBM scores, and the reported model is the last checkpoint, fixed in advance.'])

# =====================================================================================================================
K.page()
h1('3. Results')
h2('3.1 Experiments trained on MPtrj only')
fig(os.path.join(FIG, 'mptrj_runs.png'),
    'Per-epoch errors on the MPtrj validation set (20,000 structures). The 5 Å experiments and the 7 Å experiments. The brown dotted '
    'line is the final model, whose learning rate was reduced linearly from epoch 20.')
tb([['Experiment', 'Params', 'Epoch', 'E MAE', 'F MAE', 'F RMSE', 'Stress RMSE', 'WBM 3k (MAE / F1)'],
    ['5 Å, full element tables', '45.9M', '40', '15.1', '43.4', '107.0', '1.280', '0.0419 / 0.749'],
    ['5 Å, rank 32', '17.8M', '20', '17.7', '45.8', '107.1', '1.283', '-'],
    ['5 Å, rank 32, LR 0.001', '17.8M', '15', '20.2', '51.6', '118.7', '1.370', '-'],
    ['5 Å, 64 channels, rank 32', '4.7M', '13', '23.6', '53.3', '119.2', '1.393', '-'],
    ['5 Å, rank 32, DeNS', '17.8M', '17', '23.9', '55.1', '152.5', '1.373', '-'],
    ['7 Å, Adam + plateau', '17.8M', '10', '25.9', '59.9', '124.7', '1.385', '0.0635 / 0.604'],
    ['7 Å, SOAP, loss 4 : 1 : 1', '17.8M', '22', '15.0', '45.4', '101.5', '1.255', '0.0426 / 0.735'],
    [B('7 Å, SOAP, 20 epochs + 20 decay epochs'), B('17.8M'), B('40'), B('13.7'), B('42.6'), B('100.9'), B('1.238'),
     B('0.0389 / 0.775')]],
   widths=[0.27, 0.09, 0.07, 0.09, 0.09, 0.10, 0.11, 0.18],
   caption='MPtrj-only experiments. Errors are the lowest validation values each experiment reached (E: meV/atom, F: meV/Å, '
           'stress: GPa). The bold row is the final MPtrj-only model. Epoch counts differ, so this is not a '
           'like-for-like comparison.', align_right_from=1, font=8.2)
bl(['Even with the element tables factorised to rank 32, the validation error at the same epoch was 2~7 % lower. The number of '
    'parameters dropped to 39 %.',
    'With 64 channels the error was 4~8 % worse, with DeNS 13~16 %, with learning rate 0.001 2~6 %.',
    'With the cutoff extended to 7 Å and the loss settings of BAM-MP-DeNS, all metrics were better than in the 5 Å experiments at the '
    'same epoch.',
    'Adam (LR 0.001, plateau decay) fell far behind SOAP at the same epoch and was stopped at 10 epochs.',
    'A 4-fold energy weight cuts the validation energy error by 14 % without changing the WBM result.',
    'In the learning-rate decay phase the validation error stayed almost the same, but the negative bias on WBM shrank from −0.021 '
    'to −0.008 eV/atom and F1 rose from 0.724 to '
    '0.775.'])
p('For the final MPtrj-only model the validation loss stalled while the training loss kept falling (at epoch 39: training 0.00088, '
  'validation 0.00375). We read this as meaning that the limit is the amount of MPtrj data rather than the model capacity, and this '
  'is why we moved on to OMat24 pretraining.')

K.need(330)
h2('3.2 OMat24 pretraining')
fig(os.path.join(FIG, 'pretrain.png'),
    'Validation error of the OMat24 pretraining (20,000 OMat24 validation structures). One virtual epoch is 5,000 steps. The blue '
    'shading is the learning-rate decay phase.',
    width=0.92)
bl(['The 100.2 million structures of the OMat24 training set were seen about twice (58 virtual epochs, 19 hours). The run started '
    'on 5 nodes and grew to 6. One epoch takes about 9 hours on '
    '6 nodes (24 GPUs).',
    'At constant learning rate the force error fell slowly and the energy error fluctuated within 15~17 meV/atom. '
    'Once the learning rate was reduced, the energy error dropped to '
    '12.1 meV/atom.',
    'The pretrained-only model has the OMat24 energy reference, so its WBM energy metrics are meaningless. Structural '
    'accuracy was already high (3k sample RMSD '
    '0.067~0.069 Å; MPtrj-only gives 0.078 Å).'])

K.need(360)
h2('3.3 MPtrj + sAlex fine-tuning')
fig(os.path.join(FIG, 'finetune.png'),
    'Validation error of the fine-tuning (top, MPtrj validation set) and WBM 3k sample metrics (bottom). The beige band is the first '
    '10 virtual epochs, in which the 8 MPtrj copies were read clustered, and the blue band is the learning-rate decay phase. The grey '
    'dotted line is the final MPtrj-only model.')
tb([['Virtual epoch', 'Data epochs', 'Evaluation', 'F1', 'MAE', 'R<super>2</super>', 'RMSD', 'κ SRME', 'CPS'],
    ['4 (before fix)', '0.7', 'full, float32', '0.844', '0.0267', '0.857', '0.0659', '-', '-'],
    ['6 (before fix)', '1.0', 'full, float32', '0.842', '0.0269', '0.853', '-', '-', '-'],
    ['18', '3.0', 'full float32, κ float64', '0.886', '0.0225', '0.865', '0.0652', '0.220', '0.856'],
    ['22', '3.6', 'full, float64', '0.887', '0.0220', '0.869', '0.0644', '0.220', '0.857']] +
   ([] if FINAL['interim'] else [[B(str(FINAL['epoch']) + ' (final)'), B(FINAL.get('data_epochs', '6.2')), B('full, float64')] +
                                 [B(x) for x in oam]]),
   widths=[0.13, 0.11, 0.24, 0.08, 0.10, 0.08, 0.10, 0.09, 0.07],
   caption='Full-WBM evaluation (unique prototypes) during fine-tuning and of the final model. The first two rows are from the '
           'phase in which the data were read clustered.', align_right_from=3)
bl(['At virtual epoch 10, where the data reading was fixed, the validation error and the WBM metrics improved in a step. The negative '
    'bias shrank from −0.012 to −0.005 '
    'eV/atom and the precision rose from 0.78 to 0.86.',
    'After that, in the constant-learning-rate phase, the WBM metrics barely moved (3k sample MAE 0.0197~0.0199). Fine-tuning '
    'saturates at about 3 data epochs.',
    'κ SRME is 0.220, better than that of the MPtrj-only model (0.375).'] +
   ([f"The last row of this table ({FINAL['label']}) is the most recent evaluation so far. The results of the final checkpoint (virtual epoch 38) will be added after the training ends."]
    if FINAL['interim'] else
    [f"The final model after the learning-rate decay has MAE {FINAL['mae']:.4f}, F1 {FINAL['f1']:.3f} and CPS {FINAL['cps']:.3f}."]))

K.need(300)
h2('3.4 Formation energy comparison')
fig(os.path.join(FIG, 'parity.png'),
    'WBM unique-prototype formation energy: DFT (MP2020 corrected) vs model. Left MPtrj-only, right OAM; colour: '
    'point density (log). The figure\'s '
    'R<super>2</super> (formation energy) exceeds the leaderboard R<super>2</super> '
    '(energy above the convex hull).')

K.need(380)
h2('3.5 Comparison with leaderboard models')
LB = [['DPA-4.1-Max-OAM-PFT', '0.935', '0.017', '0.872', '0.058', '0.081', '0.913'],
      ['EquiformerV3+DeNS-OAM', '0.931', '0.018', '0.868', '0.059', '0.118', '0.902'],
      ['GRACE-3L-OAM-L', '0.925', '0.018', '0.875', '0.058', '0.121', '0.900'],
      ['TACE-OAM-L', '0.910', '0.020', '0.868', '0.061', '0.126', '0.889'],
      ['eSEN-30M-OAM', '0.925', '0.018', '0.866', '0.061', '0.170', '0.888'],
      ['Nequip-OAM-XL', '0.906', '0.020', '0.872', '0.063', '0.125', '0.886'],
      ['MACE-SOYO-OAM', '0.903', '0.020', '0.868', '0.062', '0.184', '0.873'],
      ['SevenNet-Omni-i12', '0.906', '0.021', '0.868', '0.062', '0.192', '0.873'],
      ['Nequip-OAM-L', '0.893', '0.022', '0.865', '0.065', '0.166', '0.870'],
      ['GRACE-2L-OAM-L', '0.883', '0.022', '0.862', '0.064', '0.169', '0.865'],
      ['Allegro-OAM-L', '0.895', '0.022', '0.868', '0.065', '0.319', '0.840'],
      ['GRACE-2L-OAM', '0.880', '0.023', '0.862', '0.067', '0.294', '0.837'],
      ['GRACE-1L-OAM', '0.824', '0.031', '0.842', '0.072', '0.517', '0.761'],
      ['MACE-SOYO-MPtrj', '0.777', '0.041', '0.799', '0.083', '0.350', '0.763'],
      ['Eqnorm MPtrj', '0.786', '0.040', '0.799', '0.084', '0.408', '0.756'],
      ['Nequip-MP-L', '0.761', '0.043', '0.791', '0.086', '0.466', '0.730']]
ours = [[B('RACE v2 OAM')] + [B(x) for x in oam],
        [B('RACE v2 MPtrj-only')] + [B(x) for x in ['0.777', '0.040', '0.800', '0.0785', '0.375', '0.761']],
        [B('BAM-MP-DeNS')] + [B(x) for x in ['0.761', '0.043', '0.797', '0.0832', '0.766', '0.672']],
        [B('BAM-OAM'), '-', B('0.027'), '-', '-', '-', '-']]
rows = LB + ours[:3]
rows.sort(key=lambda r: -float(r[6].replace('<font name="NanumB">', '').replace('</font>', '')))
rows.append(ours[3])
tb([['Model', 'F1', 'MAE', 'R<super>2</super>', 'RMSD', 'κ SRME', 'CPS']] + rows,
   widths=[0.34, 0.10, 0.11, 0.10, 0.11, 0.12, 0.12],
   caption='Part of the Matbench Discovery leaderboard (values read on 2026-11-15, unique prototypes) and the lab models. Sorted by '
           'CPS, with the lab models in bold. BAM-OAM is placed at the bottom because only its MAE is known.',
   align_right_from=1, font=8.2)
bl(['The energy and structure metrics of the OAM model are on the level of GRACE-2L-OAM-L and Nequip-OAM-L. Those models have around '
    '26 million parameters.',
    'The gap to the top models is about 0.04 in F1, 0.004~0.005 eV/atom in MAE and 0.1 in κ SRME. The top models were trained on '
    'OMat24 for 4~6 epochs or more.',
    'The MPtrj-only model sits at almost the same position as MACE-SOYO-MPtrj.'])

h2('3.6 PyTorch conversion')
bl(['JAX checkpoints can be converted with an inference-only PyTorch implementation (race_v2_torch, old name racev2_torch). The '
    'conversion works without JAX and the model file is 142 MB.',
    'The 5 Å model agrees with JAX (float64) at the 10<super>−14</super> level, checked down to per-layer outputs.',
    'After conversion, the OAM model (7 Å, trained in float32) differed from JAX (float32) in energy by at most '
    '3 × 10<super>−4</super> eV per atom on 12 equilibrium structures, '
    'the noise level of float32 arithmetic.',
    'The final OAM model was compared with JAX float64 inference on structures away from equilibrium (16 from PhononDB, atoms '
    'displaced by 0.05 Å, maximum force 1.34 eV/Å). '
    'The energy difference is at most 4 × 10<super>−7</super> eV and the force difference at most 2 × 10<super>−6</super> eV/Å. '
    'Why this is larger than the 10<super>−14</super> of the 5 Å model '
    '(possibly a float32 constant left on one side) has not been checked.'])

K.need(420)
h2('3.7 Uncertainty and out-of-distribution (OOD) detection')
bl(['A SOAP-eigenbasis Laplace approximation was applied to the final OAM model. The curvature (GGN) was estimated from 300 MPtrj and '
    'sAlex training structures, and the standard deviations (σ) of energy and forces were obtained from 16 weight samples. '
    'The prior precision λ is 1000.',
    'In-distribution (ID) sets: MPtrj and sAlex validation. Out-of-distribution (OOD) sets: OMol25 '
    '(molecules), OPoly26 (polymers) and OC25 (surface catalysis). Each set has '
    '1,000 structures; WBM uses the 3k sample.'])
fig(os.path.join(FIG, 'ood_detection.png'),
    'a: distribution of the OOD score (maximum of the per-atom force σ). b: ROC curves of ID versus OOD. c: σ against the predicted '
    'energy. Solid lines are ID, dashed lines OOD and the dotted line WBM.')
tb([['Score', 'OMol25', 'OPoly26', 'OC25', 'WBM'],
    ['Maximum of the force σ (σ_F,max)', '0.960', '0.956', '0.960', '0.396'],
    ['Mean of the force σ (σ_F)', '0.934', '0.896', '0.830', '0.381'],
    ['Energy σ per atom (σ_E)', '0.077', '0.030', '0.136', '0.601'],
    ['Total energy σ (σ_E × number of atoms)', '0.756', '0.977', '0.973', '0.268']],
   widths=[0.40, 0.15, 0.15, 0.15, 0.15],
   caption='AUROC: the probability that, for a pair of one ID structure (MPtrj validation + sAlex validation) and one structure of '
           'a given set, the score of the latter is larger. 0.5 means no separation, '
           '1.0 perfect separation.', align_right_from=1)
bl(['The force uncertainty picks out the molecule, polymer and surface data with an AUROC of 0.96. It can be used to filter out kinds '
    'of structures that were not in the training data.',
    'The energy σ per atom comes out reversed (AUROC 0.03~0.14). The OOD structures are larger (47~140 atoms versus 10~22), so '
    'averaging per atom makes σ smaller. '
    'With the total energy σ, multiplied by the number of atoms, the AUROC is 0.76~0.98.',
    'WBM is not separated from the ID sets (0.40). This is the expected result, since WBM consists of inorganic crystals of the same '
    'kind as the training data.'])
K.need(520)
fig(os.path.join(FIG, 'ood_error.png'),
    'a, b: predicted σ versus actual error (forces, energy). The black dashed line is error = σ, and the coloured lines in b are '
    'quantile bands of the normal distribution. c: energy calibration curves (solid: σ as it is; '
    'dashed: σ multiplied by a single constant). d: mean and median error of the WBM structures split into ten bins in order of σ_E.',
    width=0.86)
bl(['The absolute size of σ is much smaller than the actual error: 10~100 times for forces and about 16 times for energy. This is '
    'because this σ contains only the weight uncertainty and λ was not tuned. '
    'As it is, it cannot be used as an error bar.',
    'Multiplying by the constant 16.3 obtained on MPtrj validation brings the calibration curves of the ID sets close to the diagonal '
    '(fraction within 1σ 0.63~0.66, ideal value 0.68). WBM is still overconfident at '
    '0.55.',
    'For forces the ordering by σ follows that by error well (rank correlation 0.71~0.75 on the ID sets). For energy it is '
    'weak. On WBM the mean error of the 10 % with the smallest σ_E is '
    '12 meV/atom and that of the 10 % with the largest is 23 meV/atom, and the AUROC for picking out large-error structures is '
    '0.56~0.59.',
    'In other words, this method is good at telling "is this outside the training distribution" but can hardly tell "which WBM '
    'structures have a wrong energy".'])

K.need(300)
h2('3.8 Memory and speed')
tb([['Precision', 'Structure', 'Atoms', 'Edges', 'One evaluation (s)', 'Peak GPU memory (GB)', 'Result'],
    ['float32', 'PbTe (32 neighbours)', '8,000', '256,000', '0.52', '26.0', 'success'],
    ['float32', 'PbTe (32 neighbours)', '17,576', '562,432', '-', '-', 'out of memory'],
    ['float32', 'BeO (202 neighbours)', '2,048', '412,830', '0.93', '29.6', 'success'],
    ['float32', 'BeO (202 neighbours)', '4,000', '806,414', '-', '-', 'out of memory'],
    ['float64', 'PbTe (32 neighbours)', '4,096', '131,072', '0.37', '27.7', 'success'],
    ['float64', 'PbTe (32 neighbours)', '8,000', '256,000', '-', '-', 'out of memory'],
    ['float64', 'BeO (202 neighbours)', '864', '174,170', '0.37', '-', 'success'],
    ['float64', 'BeO (202 neighbours)', '2,048', '412,830', '-', '-', 'out of memory']],
   widths=[0.10, 0.20, 0.11, 0.12, 0.15, 0.19, 0.13],
   caption='Time and memory for one evaluation of energy and forces on a single GPU (GH200) (final OAM model, 7 Å). The structures are '
           'supercells with atoms displaced by 0.02 Å. '
           'The memory value is the peak usage of the process.', align_right_from=2, font=8.6)
bl(['The size that can be computed at once is set by the number of edges. The limit lies between 410,000~560,000 edges for float32 '
    'and between 170,000~260,000 edges for float64.',
    'Converted to a structure with 100 neighbours on average at 7 Å (the MPtrj average), a single GPU holds about 4,000~5,600 atoms '
    'in float32 and about 1,700~2,500 atoms in float64.',
    'The speed is about 2.0~2.3 microseconds per edge in float32 and about 2.1~2.8 microseconds in float64. With 100 neighbours this '
    'is 0.2~0.3 milliseconds per atom.',
    'Larger systems (MD) need a LAMMPS interface with domain decomposition or a scheme that splits the edges into parts for the '
    'computation.'])

# =====================================================================================================================
K.need(380)
h1('4. Limitations and next steps')
h2('4.1 Known issues')
tb([['Issue', 'Effect', 'Fix'],
    ['Pretraining is short (2 epochs)', 'κ and MAE worse than the top models (4~6 epochs)',
     'Continue training from the saved checkpoint up to 4 epochs'],
    ['Model has 17.8 million parameters', 'Capacity may become the limit even with more data',
     'Consider more channels, layers or rank after the pretraining-amount experiment'],
    ['High GPU memory per atom (7 Å, 128 channels)', 'In float64, several 300~512-atom supercells cannot be loaded at once',
     'Split batches by number of edges, measure the limit'],
    ['float32 inference gives a wrong κ', 'κ SRME 0.861 (0.220 in float64)',
     'Run phonons and tight relaxation in float64 (option added)'],
    ['x gate parameters are updated by SOAP', 'Values of elements absent from the data may drift',
     'Use Adam for all tables with an element axis'],
    ['Stress loss is a sum over components', 'Enters 6 times larger than intended',
     'Change to the component mean and retune the weight'],
    ['Steps 35 % slower after some restarts', 'Happens one time in three; normal after restarting again',
     'Check the speed for 2,000 steps right after a restart'],
    ['3k tracking sample is optimistic', 'MAE about 0.002 lower than on the full set',
     'Report final numbers on the full WBM set only']],
   widths=[0.30, 0.36, 0.34], caption='Known issues and how to fix them.', align_right_from=-1, font=8.2)

h2('4.2 Plan for the next compute window')
bl([B('Measure the effect of pretraining amount:') + ' Run the same fine-tuning from the 1-epoch and the 2-epoch OMat24 '
    'checkpoints and compare. This decides whether to '
    'extend pretraining or enlarge the model.',
    B('Extend pretraining:') + ' If an effect shows, continue to 4 epochs. One epoch is about 9 hours on 6 nodes.',
    B('Shorten fine-tuning:') + ' 3 data epochs plus learning-rate decay suffice (about 7 hours on 6 nodes).',
    B('Prepare for release:') + ' Tidy up the package name and directory layout (BAM_V2, race_v2), measure memory and speed, '
    'validate the PyTorch conversion.'])

K.need(170)
h2('4.3 Compute cost')
tb([['Training', 'Resources', 'Time', 'Notes'],
    ['MPtrj-only (40 epochs)', '16 GPUs (4 nodes)', 'about 23 hours', 'about 35 minutes per epoch, all float64'],
    ['OMat24 pretraining (2 epochs)', '20~24 GPUs', 'about 19 hours', 'about 3,300 structures per second (6 nodes)'],
    ['OAM fine-tuning (38 virtual epochs)', '24 GPUs (6 nodes)', f"about {FINAL['hours']} hours", 'including 5 restarts'],
    ['Full WBM evaluation', '8 GPUs + 1 node', 'about 50 minutes', 'relaxation 15~17 minutes, scoring and RMSD 33 minutes'],
    ['κ of 103 materials', '4 GPUs', 'about 23 minutes', 'float64, 3 large supercells computed separately']],
   widths=[0.30, 0.22, 0.16, 0.32], caption='Compute cost (KISTI HANGANG, 4 GPUs per GH200 node).', align_right_from=-1)

# =====================================================================================================================
K.need(280)
h1('Appendix. File locations')
tb([['Content', 'Location'],
    ['Training and evaluation code', 'Repository BAM_V2. Training was done with branch racev2 (package bam_omat24); on the main branch main '
     'the package name is race_v2 (the mapping between old and new names is in docs/LAYOUT.md)'],
    ['Evaluation and analysis scripts', 'scripts/eval/mbd (WBM, κ, ranking), scripts/eval/uq (Laplace OOD), scripts/eval/bench (memory and speed)'],
    ['Checkpoint backup', '/appsdata/t510a13/race_v2_backup_2026-11-15/checkpoints (MPtrj-only final, pretraining epochs 1 and 2, fine-tuning evaluation points and final)'],
    ['Size tables, configs, helper scripts', '/appsdata/t510a13/race_v2_backup_2026-11-15/{oam_sizes, configs, helpers}'],
    ['Training logs and evaluation results', '/scratch/t510a13/race_v2/work/runs (deleted after 15 days, so they must be moved)'],
    ['Equation notes, scripts and figures of this document', 'docs/results/race_v2_oam_mbd/{report, figures}'],
    ['Inference precision guide', 'docs/inference_precision.md'],
    ['List of next tasks', '/appsdata/t510a13/race_v2_backup_2026-11-15/NEXT_TASKS.md']],
   widths=[0.30, 0.70], caption=None, align_right_from=-1, font=8.2)
note('The equations were transcribed directly from the training code, and the parameter count computed from them '
     'matches the 17,757,551 in the log. Normalisation constants inside the libraries '
     '(tensor-product path normalisation, spherical-harmonics convention) were not checked separately. The leaderboard values were '
     'read on 2026-11-15.')

K.build(OUT, title='RACE v2: Methods and Matbench Discovery Results', author='BAM / RACE v2',
        footer='RACE v2: Methods and Matbench Discovery Results · 2026-11-16')
print(OUT)
