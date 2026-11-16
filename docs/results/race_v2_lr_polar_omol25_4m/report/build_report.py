"""RACE v2 LR-POLAR: methods (with equations) and results of the OMol25 4M run -- PDF.

usage: python -I build_report.py <font_dir> <fig_dir> <out.pdf>
Equations follow the code the model was trained with (BAM_V2_run_v3, commit d90bdc5); see methods_formulas.md.
Numbers: final test of 2026-11-16 (final_results_notes.md).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdfkit import Kit                                                                  # noqa: E402

FONT_DIR, FIG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
K = Kit(FONT_DIR, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets'))
p, h1, h2, h3, eq, bl, tb, fig, note = K.p, K.h1, K.h2, K.h3, K.eq, K.bullets, K.table, K.figure, K.note

# =====================================================================================================================
K.title('RACE v2 LR-POLAR: methods and OMol25 4M training results',
        '2026-11-16 · model RACE_POLAR (19,919,663 parameters) · training code BAM_V2_run_v3 (commit d90bdc5) · '
        'KISTI HANGANG, 12 GPUs, 170 epochs, 21.1 hours')

h1('Summary')
p('RACE v2 LR-POLAR is an interatomic potential that puts explicit physical terms on top of an equivariant '
  'message-passing backbone. The charges are obtained by charge equilibration (Qeq), but they start from per-molecule '
  'formal charges and are allowed to flow only between nearby atom pairs, and the electrostatic, polarisation, dispersion '
  'and spin energies are each computed as a separate term. This document writes down the implemented method with the '
  'equations exactly as they are in the code, and reports the results of 170 epochs of training on OMol25 4M.')
tb([['Item', '70 epochs', 'Final (170 epochs)', 'Comparison'],
    ['Internal validation, 6,000 structures (E meV / F meV/Å)', '136.7 / 26.0', '90.9 / 24.2', 'earlier run v2local: 188 / 24.8'],
    ['neutral_val test (E / F)', '-', '62.2 / 20.7', 'epoch 16: 138.5 / 28.2'],
    ['Official val sample, 32,000 structures (E / F)', '448.7 / 24.8', '363.8 / 23.3', 'UMA-s-1.2.1: 44.9 / 7.9'],
    ['Ion-water curve MAE (kcal/mol)', '0.70~1.93', '0.33~1.04', 'UMA, eSEN: 2~3 (12~20 for F<super>-</super>)'],
    ['Salt bridge MAE (kcal/mol)', '0.74', '0.48', ''],
    ['Size-consistency error, 6 Å (kcal/mol)', 'within ±0.5', 'within ±0.6', 'broken in models that take the total charge as input']],
   widths=[0.36, 0.17, 0.19, 0.28], caption='Key results. E is the energy MAE and F the force MAE.', align_right_from=1)
bl(['<font name="NanumB">What works:</font> the ion-water interaction curves are more accurate than those of the public '
    'models compared, and the energy of separated fragments converges exactly to classical electrostatics. The energy step '
    'that was seen at 4.5 Å is gone.',
    '<font name="NanumB">What is still lacking:</font> the mean error on the official val set is about 8 times that of the '
    'public models. The training data is 4M (4 % of the whole), and the error is large for highly charged structures and for '
    'metal and open-shell structures. The noble-gas and LiF dissociation curves are wrong.',
    '<font name="NanumB">Next steps:</font> train on the full OMol25 (100 million structures), OPoly26 and OC25. The '
    'preprocessing was finished or is in progress in this compute window.'])

# =====================================================================================================================
h1('1. Model overview')
p('The total energy is the sum of per-atom energies, and the forces are obtained by one automatic differentiation of the '
  'total energy with respect to the coordinates. The linear system that gives the charges and the iterative solution for '
  'the induced dipoles are both included in the differentiation, so the forces are exactly consistent with the energy.')
eq(r'E=\sum_i\left[s\left(e^{\mathrm{rep}}_i+\sum_{t=0}^{4}\left(e^{\mathrm{node}}_{t,i}+e^{x}_{t,i}+e^{f}_{t,i}\right)\right)+b\right]'
   r'+\sum_i\left(e^{\mathrm{Qeq}}_i+e^{\mathrm{spin}}_i+e^{\mathrm{elec}}_i+e^{\mathrm{disp}}_i\right)+\sum_i E^{0}_{Z_i}')
eq(r'\mathbf{F}_i=-\dfrac{\partial E}{\partial \mathbf{r}_i}')
tb([['Term', 'Content', 'Range', 'Section'],
    ['e<super>rep</super>', 'Short-range pair repulsion of UFF form', '6 Å neighbours', '2.1'],
    ['e<super>node</super>, e<super>x</super>, e<super>f</super>', 'Short-range atomic energies read out at every layer (5 layers)', '6 Å neighbours, 5 passes', '2.1'],
    ['e<super>Qeq</super>', 'On-site terms of charge equilibration (electronegativity, hardness)', 'whole structure', '2.2~2.3'],
    ['e<super>spin</super>', 'Spin-polarisation energy (local moments and pair correlation)', '6 Å neighbours', '2.6'],
    ['e<super>elec</super>', 'Coulomb energy of charges, dipoles and quadrupoles, and induced polarisation', 'whole structure (no cutoff)', '2.8'],
    ['e<super>disp</super>', 'C6 dispersion', 'whole structure (no cutoff)', '2.9'],
    ['E<super>0</super>', 'Per-element reference energy (fixed values)', '-', '2.10']],
   widths=[0.17, 0.47, 0.24, 0.12], caption='Terms that make up the total energy. s and b are learned scalars (initial values 1 and 0).',
   align_right_from=3)
p('The units are Å, e and eV, and the Coulomb constant is K = 14.3996 eV·Å. In this document S(r; a, b) is a smooth '
  'switching function that is 0 for r ≤ a and 1 for r ≥ b.')
eq(r'S(r;a,b)=x^{3}\left(10-15x+6x^{2}\right),\qquad x=\mathrm{clip}\left(\dfrac{r-a}{b-a},\,0,\,1\right)')

# =====================================================================================================================
h1('2. Methods')
h2('2.1 Short-range backbone')
p('On a graph with atoms as nodes and atom pairs within 6 Å as edges, equivariant messages are passed 5 times. The node '
  'features are spherical tensors with l = 0~3 (128 channels per type). The distance inputs are 12 Bessel functions and 16 '
  'exponential functions built from atomic shell radii, multiplied by the polynomial cutoff function f<sub>cut</sub>.')
eq(r'B_n(r)=\dfrac{2}{r_c}\,\dfrac{\sin(n\pi r/r_c)}{r}\ \ (n=1,\ldots,12),\qquad '
   r'f_{\mathrm{cut}}(r)=1-28u^{6}+48u^{7}-21u^{8},\ \ u=\dfrac{r}{r_c},\ \ r_c=6\ \mathrm{\AA}')
eq(r's_{ab}(r)=\exp\left(-\dfrac{r}{\rho_a(Z_i)+\rho_b(Z_j)}\right)f_{\mathrm{cut}}(r),\qquad a,b\in\{s,p,d,f\}')
p('The features are updated from the message m of layer t, x built from the atomic basis A, and their tensor product p. '
  'Dividing by the average number of neighbours (32.58) sets the scale.')
eq([r'A_i=\dfrac{1}{\bar{n}}\sum_{j\in N(i)}R^{(0)}(r_{ij})\odot\left[W^{(0)}_1 e_j\otimes Y(\hat{\mathbf{r}}_{ij})\right],\qquad '
    r'm_i=\dfrac{1}{\bar{n}}\sum_{j\in N(i)}R^{(t)}(r_{ij})\odot\left[W_1\tilde{h}_j\otimes Y(\hat{\mathbf{r}}_{ij})\right]',
    r'x_i=\hat{W}_t A_i,\qquad p_i=W_2\left[(W_m m_i)\otimes x_i\right],\qquad \delta_i=\gamma\odot\left(p_i+S_{Z_i}\tilde{h}_i\right)'])
eq(r'h^{(t+1)}=\delta\ \ (t=0,1),\qquad h^{(t+1)}=2\delta-h^{(t-1)}\ \ (t\geq 2)')
bl(['Y are the real spherical harmonics, R<super>(t)</super> is a distance MLP (28 → 64 → 64 → 64) learned separately for each layer, and e<sub>j</sub> is the element embedding.',
    '<font name="NanumB">Spectral normalisation:</font> the weights that build x are divided, block by block, by their largest '
    'singular value (Ŵ). This prevents the problem that the weak signal of distant neighbours was amplified and produced a step at the cutoff distance.',
    '<font name="NanumB">Chebyshev residual:</font> from the third layer on, a recurrence of the form h<super>(t+1)</super> = 2δ − h<super>(t−1)</super> is used.',
    'The per-element weight tables (S<sub>Z</sub> etc.) are factorised with rank 32, so the 94 elements share parameters.'])
p('In each layer three scalar energies are read out and added. LN is layer normalisation, and the normalisation constant of the x readout is (3×10<super>−3</super>)<super>2</super>.')
eq(r'e^{\mathrm{node}}_{t,i}=w\cdot\mathrm{gate}\left(W_r h^{(t+1)}_i\right),\qquad '
   r'e^{x}_{t,i}=\mathrm{MLP}^{x}_{Z_i}\left(\dfrac{\mathrm{LN}(x_i)}{\sqrt{\bar{n}}}\right),\qquad '
   r'e^{f}_{t,i}=\mathrm{MLP}^{f}_{Z_i}\left(\dfrac{\mathrm{LN}(\tilde{h}^{(t)}_i)}{\sqrt{\bar{n}}}\right)')
p('Repulsion at very short distances is rare in the training data, so a pair potential built from the values of the UFF table was added.')
eq(r'e^{\mathrm{rep}}_i=\dfrac{1}{2}\sum_{j\in N(i)}D_{ij}\,\dfrac{6}{\zeta_{ij}-6}\,'
   r'\exp\left[\zeta_{ij}\left(1-\dfrac{r_{ij}}{x_{ij}}\right)\right]f_{\mathrm{cut}}(r_{ij})\,'
   r'\mathrm{sigmoid}\left(\dfrac{r_{ij}-1.25\,(R_i+R_j)}{0.3\ \mathrm{\AA}}\right)')

h2('2.2 Charge equilibration (Qeq)')
p('The neural network predicts an electronegativity χ and a hardness J for every atom, and the charges are obtained by '
  'minimising the energy functional defined by these values. Each atomic charge is treated as a Gaussian distribution of width σ.')
eq([r'\chi_i=\chi^{0}_{Z_i}+z_{i,1},\qquad J_i=\mathrm{softplus}\left(J^{0}_{Z_i}+z_{i,2}\right),\qquad '
    r'\sigma_i=r^{\mathrm{cov}}_{Z_i}\left[0.5+1.5\,\mathrm{sigmoid}(u_{Z_i})\right]',
    r'A_{ii}=J_i+\dfrac{K}{\sqrt{\pi}\,\sigma_i},\qquad '
    r'A_{ij}=K\,\dfrac{\mathrm{erf}\left(r_{ij}/\sqrt{2(\sigma_i^{2}+\sigma_j^{2})}\right)}{r_{ij}}'])
eq(r'E_{\mathrm{solve}}(\mathbf{q})=\sum_i\chi_i q_i+\dfrac{1}{2}\sum_{i,j}q_i A_{ij}q_j')
bl(['z<sub>i</sub> is obtained from the scalar features of the last layer by an MLP (128 → 32 → 2). χ<super>0</super> is initialised with the Pauling electronegativity.',
    '<font name="NanumB">Bounded width:</font> σ is learned per element only and is bounded to 0.5~2 times the covalent radius. '
    'In earlier experiments the width grew without bound and the charges became independent of the structure.',
    'The matrix A is built for all atom pairs in the structure, without a cutoff. Each structure is put in a block of 352 slots and the charges are obtained with one dense linear solve.',
    'Only the on-site terms of this functional enter the energy. The pair Coulomb energy is counted only once, in the electrostatics block of Section 2.8.'])
eq(r'e^{\mathrm{Qeq}}_i=\chi_i q_i+\dfrac{1}{2}\left(J_i+\dfrac{K}{\sqrt{\pi}\,\sigma_i}\right)q_i^{2}')

h2('2.3 Conductance-based charge transfer')
p('Ordinary Qeq redistributes charge freely over the whole structure, so charge flows even between distant molecules and fractional charges remain in the dissociation limit. '
  'In LR-POLAR the charges start from the per-molecule formal charges q<super>0</super> and move only along nearby pairs that have a conductance c<sub>ij</sub>.')
eq([r'\mathbf{q}=\mathbf{q}^{0}+B\mathbf{p},\qquad \mathbf{p}=\mathrm{argmin}_{\mathbf{p}}\left[E_{\mathrm{solve}}(\mathbf{q}^{0}+B\mathbf{p})'
    r'+\dfrac{1}{2}\sum_{\mathrm{pairs}}\dfrac{p_{ij}^{2}}{c_{ij}}\right]',
    r'\Longleftrightarrow\quad (I+LA)\,\mathbf{q}=\mathbf{q}^{0}-L\boldsymbol{\chi},\qquad L=\mathrm{diag}\left(\sum_j c_{ij}\right)-c'])
p('p<sub>ij</sub> is the charge moved across the pair, and moving it carries a penalty of p<super>2</super>/2c. The conductance is set by two per-element parameters (g, b).')
eq([r'c_{ij}=\exp\left[s_{ij}-\beta_{ij}\,(r_{ij}-r_0)\right]\left[1-S(r_{ij};3.0,4.5)\right],\qquad r_0=2\ \mathrm{\AA}',
    r's_{ij}=\ln C-\mathrm{softplus}\left(\ln C-g_{Z_i}-g_{Z_j}\right),\qquad '
    r'\beta_{ij}=\beta_{\min}+\mathrm{softplus}\left(b_{Z_i}+b_{Z_j}\right)'])
bl(['C = 100 e<super>2</super>/eV is the upper limit of the conductance and β<sub>min</sub> = 2.5 Å<super>−1</super> the lower limit of the decay rate. The starting value is β = 4 Å<super>−1</super>.',
    '<font name="NanumB">What is conserved:</font> the column sums of L are 0, so the sum of q<super>0</super> is conserved exactly in every group of atoms connected by edges shorter than 4.5 Å. '
    'An atom without conductance keeps its q<super>0</super>, and when the conductance is very large the charges converge to the Qeq charges within that group.',
    '<font name="NanumB">Learned decay:</font> without decay, charge transfer switched on abruptly at 4.5 Å and produced a step in the ion-water energy. '
    'It was removed by reducing the conductance exponentially with distance and learning the decay rate (Figure 4).',
    'The matrix I + LA is not symmetric, so it is solved with dense LU, and it is computed in float64 so that charge conservation is not broken.'])
note('Caution: the decay factor is multiplied after the upper limit, so for r &lt; 2 Å the conductance exceeds the upper limit C. No problem was seen in the results, but it is an item to fix in the next training.')

h2('2.4 Formal charge and formal spin')
p('A molecule is defined as a group of atoms connected by bonds (pairs whose distance is shorter than 1.2 times the sum of the covalent radii). '
  'The integer charge of molecule m is determined and divided evenly among the atoms of that molecule; this is q<super>0</super>.')
eq(r'Q_m=\mathrm{round}\left(\sum_{i\in m}v_i\right),\qquad q^{0}_i=\dfrac{Q_{m(i)}}{N_{m(i)}},\qquad \sum_m Q_m=Q')
p('For the per-atom value v, the first one available in the order below is used. If the sum of the molecular charges differs from the total charge Q, they are corrected by ±1 starting from the molecule with the largest remainder.')
tb([['Order', 'Source', 'Used when'],
    ['1', 'Per-atom formal charges given by the user', 'inference, special cases'],
    ['2', 'NBO charge labels, or Löwdin charge labels if absent', 'training (most rows of OMol25)'],
    ['3', 'Automatic assignment by bonding rules', 'inference only (when there are no labels)'],
    ['4', 'Q or 0 when there is a single molecule or it is neutral', 'simple cases without labels'],
    ['5', 'Q/N distributed uniformly over all atoms', 'last resort (breaks size consistency)']],
   widths=[0.1, 0.5, 0.4], caption='Order in which the formal charge q<super>0</super> is determined.', align_right_from=-1)
p('The formal spin s<super>0</super> is handled the same way. With one molecule, 2S = multiplicity − 1 is used; with several, the per-molecule sums of the Löwdin spin labels are '
  'rounded consistently with the parity of the number of electrons.')
h3('Automatic assignment (inference only)')
p('For structures without labels the formal charges are determined by bonding rules. The order is as follows.')
bl(['Find the bonds; for metals, cut the bonds and give a fixed ionic charge or the default oxidation state.',
    'Compare valence and number of bonds to determine onium (+1) and anion sites, and place π bonds by maximum matching on short bonds (shorter than 0.97 times the sum of the covalent radii).',
    'Apply the hypervalence of S, P and halogens and onium rules such as nitro, pyridinium and guanidinium.',
    'Treat the remaining valences as lone-pair anions, carbanions next to a metal, or radicals.',
    'Finally, choose the ambiguous sites (isolated H, halides, metals) by an exact combinatorial search so that the total charge and the multiplicity are matched.'])
p('On a sample of training shards it agreed with the q<super>0</super> obtained from the labels for 90.1 % of the rows, and on the official val sample for 89.97 %. On the first 200 of the official val set, '
  'using automatic assignment instead of uniform distribution reduced the energy MAE from 486.5 to 274.2 meV (at 70 epochs).')

h2('2.5 Local charge / spin inputs and size consistency')
p('To treat systems in different charge or spin states with one model, that information has to be given as input. Feeding the total charge Q and spin S of the structure '
  'identically to all atoms (the public model UMA does the same) breaks size consistency: when one distant ion is added, Q changes and even the energy of the other molecules '
  'changes. LR-POLAR gives every atom only the sum of the formal charges and formal spins within 6 Å around it.')
eq(r'\bar{Q}_i=q^{0}_i+\sum_{j\in N(i)}w(r_{ij})\,q^{0}_j,\qquad 2\bar{S}_i=s^{0}_i+\sum_{j\in N(i)}w(r_{ij})\,s^{0}_j,\qquad '
   r'w(r)=1-S(r;4.5,6.0)')
eq(r'h^{(0)}_i=e_i+\left[W_{c2}\,\mathrm{silu}\left(W_{c1}[\bar{Q}_i,2\bar{S}_i]+T_q(\bar{Q}_i)+T_s(2\bar{S}_i)\right)+b\right]\left(1+G_{Z_i,0}\right)')
bl(['T<sub>q</sub> and T<sub>s</sub> are learned tables with one row per integer charge / spin, linearly interpolated between rows. The same hidden value is also added to the '
    'scalar features of the 1st~4th layers, through per-layer weights and a per-element gate G.',
    '<font name="NanumB">Size consistency:</font> a fragment farther away than 6 Å does not enter the input, so the energy of one fragment is independent of the '
    'presence of other fragments. For a small molecule with all atoms within 4.5 Å it becomes the same as a model that takes the total charge.',
    'All related weights are initialised to 0, so at the start of training the model is the same as one without these inputs.'])

h2('2.6 Spin head and spin energy')
p('The spin energy is written with two invariants: the local moment of an atom, μ ≥ 0, and the correlation of a neighbour pair, c ∈ [−1, 1]. μ corresponds to the number of unpaired electrons.')
eq(r'\mu_i=u_g\left[\mu^{\mathrm{raw}}_i+w_i\max\left(0,\ 2S-\sum_k\mu^{\mathrm{raw}}_k\right)\right],\qquad '
   r'\mu^{\mathrm{raw}}_i=\mathrm{softplus}(\cdot)')
eq(r'E_{\mathrm{spin}}=\sum_i\left[\Delta_i\mu_i+\dfrac{1}{2}W_{Z_i}\mu_i^{2}\right]+\sum_{i<j}J_{ij}\,\mu_i\mu_j\,\dfrac{c_{ij}+c_{ji}}{2}')
eq(r'J_{ij}=A_{Z_iZ_j}\exp\left[-(\kappa_{Z_i}+\kappa_{Z_j})\left(r_{ij}-r^{\mathrm{cov}}_i-r^{\mathrm{cov}}_j\right)\right]\left[1-S(r_{ij};4,6)\right]')
bl(['Δ (on-site energy) and c come from the neural network; W (per-element spin coupling constant), κ and A are per-element parameters. W was initialised for H, C, N and O with '
    'the values of spin-polarised DFTB (−1.95, −0.62, −0.71, −0.76 eV).',
    '<font name="NanumB">Sum condition:</font> the sum of μ cannot be smaller than 2S. If it falls short, the deficit is distributed with softmax weights w.',
    '<font name="NanumB">Restricted / unrestricted:</font> a singlet obtained from a restricted calculation has u<sub>g</sub> = 0, so μ = 0 and E<sub>spin</sub> = 0. '
    'Open-shell states and unrestricted (broken-symmetry) singlets have u<sub>g</sub> = 1. The two solutions of the same structure can thus be distinguished in training.',
    'As an auxiliary output the signed spin density m<sub>i</sub> is also predicted. It is corrected so that the sum is exactly 2S, and it does not enter the energy.'])
eq(r'm_i=m^{\mathrm{raw}}_i+w_i\left(2S-\sum_k m^{\mathrm{raw}}_k\right)')
note('Caution: max(0, ·) creates a kink where the derivative is discontinuous. Also, this correction and u<sub>g</sub> use values of the whole structure, so the spin energy, unlike '
     'the local inputs of Section 2.5, is not fully size consistent.')

h2('2.7 NBO redistribution head')
p('OMol25 has two kinds of charge / spin labels, Löwdin and NBO. The charges q used for electrostatics are fitted to the Löwdin charges, and the NBO values are connected through a '
  'redistribution map between neighbours. The map conserves the sum.')
eq(r'R_t[v]_i=v_i+\sum_{j\in N(i)}t_{ij}\,(v_j-v_i),\qquad '
   r't_{ij}=t_{\max}\tanh\left(\dfrac{\mathrm{MLP}(\cdot)}{t_{\max}}\right)\left[1-S(r_{ij};3.0,4.5)\right],\quad t_{\max}=0.5')
bl(['In the loss the map is applied to the labels. That is, R[q<super>Löwdin label</super>] is compared with the NBO labels, so this term trains only the pair coefficients t.',
    'The NBO charges are used first when the per-molecule formal charge q<super>0</super> is determined, so they enter the model indirectly.'])

h2('2.8 Electrostatics and polarisation')
p('Every atom has a charge q (Section 2.3), a permanent dipole μ, a permanent quadrupole Θ and a polarisability α. μ, Θ and α are predicted from the last-layer features by equivariant linear maps. '
  'The interaction between multipoles uses a Gaussian-smeared Coulomb kernel.')
eq(r'T_a(r)=\dfrac{\mathrm{erf}(r/a)}{r},\qquad '
   r'\alpha_i=\alpha_{\min}+\left(\alpha^{\max}_{Z_i}-\alpha_{\min}\right)\mathrm{sigmoid}(\cdot)')
eq(r'a_0=2\sigma=2\ \mathrm{\AA}\ \ (\mathrm{permanent}),\qquad s_i=\alpha_i^{1/3},\qquad '
   r'a_i=\sqrt{2(s_i^{2}+\sigma^{2})},\qquad a_{ij}=\sqrt{2(s_i^{2}+s_j^{2})}')
p('The induced dipoles are solved self-consistently in the field F of the permanent multipoles. The interaction T between induced dipoles is also included.')
eq(r'\left(\dfrac{1}{\alpha_i}+\epsilon\right)\boldsymbol{\mu}^{\mathrm{ind}}_i-\left(T\boldsymbol{\mu}^{\mathrm{ind}}\right)_i=\mathbf{F}_i')
eq([r'e^{\mathrm{perm}}_i=\dfrac{K}{2}\left[q_i\phi_i-\boldsymbol{\mu}_i\cdot\mathbf{E}_i+\dfrac{1}{3}\Theta_i:G_i\right]',
    r'e^{\mathrm{pol}}_i=K\left[\dfrac{1}{2}\left(\dfrac{1}{\alpha_i}+\epsilon\right)\left|\boldsymbol{\mu}^{\mathrm{ind}}_i\right|^{2}'
    r'-\boldsymbol{\mu}^{\mathrm{ind}}_i\cdot\mathbf{F}_i-\dfrac{1}{2}\boldsymbol{\mu}^{\mathrm{ind}}_i\cdot\left(T\boldsymbol{\mu}^{\mathrm{ind}}\right)_i\right]',
    r'e^{\mathrm{elec}}_i=e^{\mathrm{perm}}_i+e^{\mathrm{pol}}_i'])
bl(['φ, <font name="NanumB">E</font> and G are the potential, field and field gradient that the charges, dipoles and quadrupoles of all other atoms create at atom i. '
    'For molecules (open boundary) all pairs in the structure are summed without a cutoff.',
    'The induced dipoles are solved with the conjugate gradient method (tolerance 10<super>−4</super>, at most 25 iterations). The forces are obtained by differentiating through the whole solve.',
    'The polarisability ranges over 0.001~4 Å<super>3</super> (up to 1 Å<super>3</super> for hydrogen) and the starting value is 0.7 Å<super>3</super>. The induced width s is '
    'smoothly bounded so that it does not exceed 1 Å.',
    'The dipole moment of the structure is Σ(q<sub>i</sub><font name="NanumB">r</font><sub>i</sub> + μ<sub>i</sub> + μ<sub>i</sub><super>ind</super>) and is '
    'fitted to the DFT dipole labels.'])
note('Caution: the configuration key is called thole_width, but the actual damping is not the exponential function of Thole; it is a Gaussian smearing whose width is proportional to α<super>1/3</super>. '
     'Also, the self energy of the charges is contained once in the Qeq on-site term (per-element width) and once in this block (width 1 Å). It is a constant difference that training absorbs, but it needs to be cleaned up.')

h2('2.9 Dispersion')
p('The C6 coefficients are built from the polarisabilities of Section 2.8. The dispersion term therefore also trains the polarisabilities, and the coordinate dependence of the polarisabilities enters the forces.')
eq([r'C_{6,ii}=\mathrm{softplus}(\kappa_{Z_i})\,\alpha_i^{2},\qquad '
    r'C_{6,ij}=\dfrac{2\,C_{6,ii}\,C_{6,jj}}{\dfrac{\alpha_j}{\alpha_i}C_{6,ii}+\dfrac{\alpha_i}{\alpha_j}C_{6,jj}}',
    r'E_{\mathrm{disp}}=-\sum_{i<j}\dfrac{C_{6,ij}}{r_{ij}^{6}+\left(R^{0}_{ij}\right)^{6}},\qquad '
    r'R^{0}_{ij}=a_1\,(R_i+R_j),\qquad R_i=1.344\ \mathrm{\AA}\left(\dfrac{\alpha_i}{0.148\ \mathrm{\AA}^{3}}\right)^{1/7}'])
bl(['The combination rule is Tkatchenko-Scheffler and the damping is of Becke-Johnson form. κ is learned per element and a<sub>1</sub> as a single scalar (starting values 7.5 eV, 0.8).',
    'For molecules all pairs in the structure are summed without a cutoff.'])

h2('2.10 Loss and label handling')
p('The energy label is the DFT energy minus the sum of the per-element reference energies. The energy residual is divided by the number of atoms. The Huber function is used for energies and forces.')
eq(r'\mathcal{L}=10\,\mathcal{L}_E+\mathcal{L}_F+0.001\,\mathcal{L}_q+1.875\times10^{-5}\,\mathcal{L}_{\mathrm{dip}}+6.25\times10^{-4}\,\mathcal{L}_{\mathrm{spin}}')
eq([r'\mathcal{L}_E=\dfrac{1}{G}\sum_g H_{0.05}\left(\dfrac{E^{\mathrm{pred}}_g-E^{\mathrm{ref}}_g}{N_g}\right),\qquad '
    r'\mathcal{L}_F=\dfrac{1}{3N}\sum_i\sum_{a=x,y,z}H_{0.25}\left(F^{\mathrm{pred}}_{i,a}-F^{\mathrm{ref}}_{i,a}\right)',
    r'H_\delta(x)=\dfrac{x^{2}}{2}\ \ (|x|\leq\delta),\qquad H_\delta(x)=\delta\left(|x|-\dfrac{\delta}{2}\right)\ \ (|x|>\delta)'])
eq([r'\mathcal{L}_q=\dfrac{1}{N}\sum_i l_i\left(q_i-q^{\mathrm{L}}_i\right)^{2}+\left\langle\left(R_{t}[q^{\mathrm{L}}]_i-q^{\mathrm{NBO}}_i\right)^{2}\right\rangle,\qquad '
    r'\mathcal{L}_{\mathrm{dip}}=\left\langle\dfrac{1}{3}\left|\mathbf{p}_g-\mathbf{p}^{\mathrm{ref}}_g\right|^{2}\right\rangle',
    r'\mathcal{L}_{\mathrm{spin}}=\left\langle\left(\mu_i-|m^{\mathrm{ref}}_i|\right)^{2}\right\rangle+\left\langle\left(c_{ij}-\mathrm{sign}(m^{\mathrm{ref}}_i m^{\mathrm{ref}}_j)\right)^{2}\right\rangle'
    r'+\left\langle\left(m_i-m^{\mathrm{ref}}_i\right)^{2}\right\rangle+\left\langle\left(R_{t}[m^{\mathrm{ref}}]_i-m^{\mathrm{NBO}}_i\right)^{2}\right\rangle'])
bl(['The Huber thresholds are 0.05 eV/atom for the energy and 0.25 eV/Å for the forces. Before the Huber function the residuals are additionally squashed with tanh (energy 5 eV/atom, forces 10 eV/Å) '
    'so that extreme structures cannot destabilise training.',
    '<font name="NanumB">Missing labels:</font> for rows without charge, spin or dipole labels the weight of that term is 0. Energies and forces are always trained.',
    'The superscript L denotes Löwdin labels, NBO denotes NBO labels, and p is the dipole moment of the structure.',
    '<font name="NanumB">Spin correlation:</font> the target of c is the product of the signs of the spin labels of the two atoms, and it is counted only when both atoms have |m| ≥ 0.1.',
    'The validation loss is computed from 10 L<sub>E</sub> + L<sub>F</sub> only, and the best checkpoint is chosen by evaluating with the exponential moving average (EMA) weights.'])
note('Caution: the terms written with ⟨·⟩ (NBO charges, dipole, spin) are averaged per GPU and then added, so the effective weight is multiplied by the number of GPUs (12). '
     'With the settings above this gives 0.012, 2.25×10<super>−4</super> and 7.5×10<super>−3</super>. Changing the number of GPUs changes the balance, so it is an item to fix in the next training.')

K.need(230)
h2('2.11 Training procedure')
tb([['Item', 'Value'],
    ['Data', 'OMol25 4M: 3,936,920 training structures (216 million atoms), 10,000 validation structures, 5,000 neutral_val test structures'],
    ['Setup', '12 GPUs (3 nodes × 4). About 556 graphs per step, 1,306 steps per epoch. 170 epochs correspond to about 32 passes over the data'],
    ['Optimiser', 'SOAP (b1 = b2 = 0.95, preconditioner updated every 10 steps). Adam for 0- and 1-dimensional parameters and per-element tables'],
    ['Learning rate', 'Maximum 0.01. Linear warm-up for 2 epochs, then cosine decay down to a factor of 0.05'],
    ['Other', 'Gradient-norm clipping 1.0, EMA 0.999 (with initial warm-up); the model is float32, the charge solve and the SOAP statistics are float64'],
    ['Cost', 'About 448 s per epoch, 21.1 hours in total (Slurm job 1280748)']],
   widths=[0.17, 0.83], caption='Training setup.', align_right_from=-1)
eq(r'\eta(t)=\eta_0\left[0.05+0.95\cdot\dfrac{1}{2}\left(1+\cos\dfrac{\pi\,(t-t_w)}{T-t_w}\right)\right],\qquad '
   r'd_t=\min\left(0.999,\ \dfrac{1+t}{10+t}\right)')
p('After training, the per-atom energy constant is refitted once on a sample of the training data (gauge correction). This time it was +0.145 meV/atom, practically 0.')

# =====================================================================================================================
K.page()
h1('3. Results')
h2('3.1 Training curves')
fig(os.path.join(FIG, 'loss_curves.png'),
    'Loss and validation error per epoch. Top left: total loss. Top right: terms of the training loss. Bottom: energy and force MAE of the stream validation (all domains) and '
    'of the neutral_val test.')
bl(['The validation loss stalled at 0.00229~0.00230 from around epoch 143 while only the training loss kept decreasing, so the gap widened. It did not rise, so the model '
    'did not get worse by overfitting, but almost all the generalisation obtainable from the 4M data appears to have been obtained.',
    'The energy improved little by little until the end (neutral_val 138.5 → 62.2 meV), and the forces hardly changed after epoch 90 (25 meV/Å).',
    'The best checkpoint is the last epoch (170, validation loss 0.002291).'])

K.need(330)
h2('3.2 Validation error')
tb([['Group', 'Structures', '70 epochs E', 'Final E', '70 epochs F', 'Final F'],
    ['All', '6,000', '136.7', '90.9', '26.0', '24.2'],
    ['Biomolecules', '1,179', '127.0', '55.4', '12.9', '11.5'],
    ['Electrolytes', '1,188', '197.1', '114.8', '21.2', '19.5'],
    ['Metal complexes', '1,153', '247.3', '204.3', '38.9', '37.0'],
    ['Reactivity', '1,218', '74.5', '61.1', '54.4', '52.0'],
    ['Neutral (q = 0)', '-', '93.7', '55.5', '21.0', '19.3'],
    ['q = ±1', '-', '160.2', '108.3', '25.4', '23.8'],
    ['|q| ≥ 2', '-', '261.1', '197.5', '39.2', '37.2'],
    ['Closed shell', '-', '112.5', '64.1', '20.3', '18.5'],
    ['Open shell', '-', '259.6', '227.1', '51.4', '49.9'],
    ['With metal', '-', '262.6', '221.6', '44.2', '42.4'],
    ['Without metal', '-', '107.6', '60.7', '20.4', '18.6']],
   widths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
   caption='Per-group error on the 6,000 internal validation structures (E: meV, F: meV/Å). Formal charges obtained from the labels are used.')
tb([['Group', 'Structures', '70 epochs E', 'Final E', '70 epochs F', 'Final F'],
    ['All', '32,000', '448.7', '363.8', '24.8', '23.3'],
    ['Biomolecules', '7,013', '326.6', '206.7', '13.6', '12.3'],
    ['Electrolytes', '19,725', '539.2', '452.7', '26.8', '25.2'],
    ['Metal complexes', '2,242', '380.2', '330.9', '55.8', '54.5'],
    ['Reactivity', '2,598', '199.1', '183.4', '68.6', '66.4'],
    ['Neutral (q = 0)', '13,572', '278.5', '192.5', '17.3', '16.0'],
    ['q = ±1', '13,844', '342.9', '262.1', '23.4', '21.8'],
    ['|q| ≥ 2', '4,584', '1272.2', '1178.1', '46.8', '45.2'],
    ['Closed shell', '29,070', '420.7', '329.5', '21.4', '19.8'],
    ['Open shell', '2,930', '726.3', '703.7', '67.4', '66.7'],
    ['<font name="NanumB">UMA-s-1.2.1 (all)</font>', '32,000', '', '44.9', '', '7.9'],
    ['<font name="NanumB">eSEN-md (all)</font>', '32,000', '', '47.9', '', '4.8']],
   widths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
   caption='Error on the 32,000-structure sample of the official OMol25 val set (E: meV, F: meV/Å). The formal charges were assigned automatically by bonding rules. The public models were evaluated on the same sample.')
bl(['The official val set is much harder than the internal validation. Large electrolyte clusters make up more than 60 % and there are many highly charged structures.',
    'The error of the |q| ≥ 2 group (1,178 meV) pulls up the overall mean. The rows where the automatically assigned formal charge is wrong (about 10 %) are suspected to be the main cause, but '
    'this was not confirmed by re-evaluating with the label formal charges.',
    'We think most of the difference from the public models comes from the amount of data. UMA and eSEN were trained on the full OMol25 (100 million structures) and LR-POLAR used only 4 %. '
    'Confirming this requires training on the full data.'])

K.need(300)
h2('3.3 Physics benchmarks')
tb([['Curve', '70 epochs', 'Final', 'DFT minimum', 'Final model'],
    ['Salt bridge (acetate…methylammonium)', '0.74', '0.48', '−109.2', '−108.9'],
    ['NaCl dissociation', '26.1', '9.5', '−98.1', '−108.7'],
    ['LiF dissociation', '12.2', '20.0', '−137.3', '−161.0'],
    ['Li<super>+</super>…I<super>−</super>', '4.0', '2.9', '−136.9', '−137.0'],
    ['CH<sub>4</sub> dimer', '0.05', '0.03', '−0.46', '−0.38'],
    ['Benzene sandwich', '0.36', '0.35', '−1.91', '−2.19'],
    ['Halogen bond (CF<sub>3</sub>Br…NH<sub>3</sub>)', '1.2', '1.2', '−3.74', '−1.84'],
    ['Ar<sub>2</sub>', '3.8', '3.0', '−0.28', '+3.66'],
    ['Li<super>+</super>…Xe', '31.0', '31.5', '−9.75', '+35.9']],
   widths=[0.36, 0.14, 0.14, 0.18, 0.18],
   caption='MAE of the interaction-energy curves and the values at the position of the DFT minimum (kcal/mol).')
fig(os.path.join(FIG, 'bench_vs_public.png'),
    'Physics benchmark curves: DFT (black dots), RACE v2 LR-POLAR final, earlier run (v2local), public models. The public models lose the interaction near 6 Å, so their '
    'ion-pair curves become flat.', width=0.92)
bl(['The long-range tail (1/r) of the ion pairs and of the salt bridge is followed accurately. This is the result of fixing the formal charges and computing the electrostatics without a cutoff.',
    'LiF got worse than in the middle of training. It overbinds by 24 kcal/mol at the minimum. The energy reference of isolated ions is suspected to be off.',
    'Noble gases (Ar<sub>2</sub>, Li<super>+</super>…Xe) repel at distances where there should be attraction. These elements are almost absent from the training data, and '
    'it did not improve as training progressed.',
    'The halogen bond is reproduced only about half.'])

K.need(260)
h2('3.4 Ion-water interaction')
tb([['Curve', 'DFT minimum', 'Model minimum', 'Error at minimum', 'MAE (1.9~10 Å)', 'Model/DFT at 10 Å'],
    ['Li<super>+</super>', '−34.39', '−34.59', '−0.18', '0.41', '0.85'],
    ['Na<super>+</super>', '−24.25', '−24.71', '−0.46', '0.60', '0.85'],
    ['K<super>+</super>', '−17.52', '−17.49', '+0.03', '0.33', '0.85'],
    ['F<super>−</super> (C<sub>2v</sub>)', '−17.47', '−16.55', '+1.93', '1.04', '0.86'],
    ['F<super>−</super> (hydrogen bond)', '−24.41', '−23.27', '+1.13', '0.56', '0.88'],
    ['Cl<super>−</super> (C<sub>2v</sub>)', '−12.48', '−13.23', '−0.75', '0.67', '0.86'],
    ['Cl<super>−</super> (hydrogen bond)', '−13.89', '−14.01', '−0.09', '0.40', '0.87']],
   widths=[0.2, 0.15, 0.15, 0.15, 0.17, 0.18],
   caption='Ion-water interaction curves (kcal/mol). The reference is a ωB97M-V/def2-TZVPD calculation.')
fig(os.path.join(FIG, 'ion_water_vs_public.png'),
    'Ion-water interaction curves: DFT compared with RACE v2 LR-POLAR final, the earlier run and public models. The bottom-right panel shows the MAE per curve.')
bl(['The MAE of LR-POLAR is 0.33~1.04 kcal/mol. The public models are at 2~3 kcal/mol, their interaction becomes 0 beyond 6 Å, and for F<super>−</super> they are '
    'wrong by as much as 12~20 kcal/mol.',
    'The 4.5~5 Å step of the earlier run is gone. In the scan at 0.05 Å spacing the largest jumps are 0.29~0.60 kcal/mol and all occur at 3.50~3.55 Å. '
    'Small kinks remain in the K<super>+</super> and F<super>−</super> curves.',
    'At 10~20 Å the interaction is 0.84~0.88 times the DFT value, about 15 % too weak. We presume that the dipole of the water molecule is predicted too small.'])

K.need(300)
h2('3.5 Size consistency')
p('Two fragments are separated by a distance R, and we look at the interaction energy of the model minus the point-charge Coulomb energy. The closer to 0 the better.')
tb([['Pair', 'R = 6 Å', '8 Å', '10 Å', '15 Å', '20 Å', 'Charge at 50 Å'],
    ['Li<super>+</super>…I<super>−</super>', '−0.60', '−0.19', '−0.08', '−0.02', '0.00', '−1.000'],
    ['Na<super>+</super>…Cl<super>−</super>', '−0.59', '−0.18', '−0.07', '−0.01', '0.00', '−1.000'],
    ['Li<super>+</super>…Xe', '−0.40', '−0.13', '−0.05', '−0.01', '0.00', '0.000'],
    ['Na<super>+</super>…H<sub>2</sub>O', '+0.29', '+0.23', '+0.17', '+0.08', '+0.05', '0.000'],
    ['Cl<super>−</super>…H<sub>2</sub>O', '−0.25', '+0.18', '+0.13', '+0.07', '+0.04', '0.000'],
    ['F<super>−</super>…H<sub>2</sub>O', '−0.33', '+0.18', '+0.13', '+0.07', '+0.04', '0.000'],
    ['Ar…Ar', '−0.02', '0.00', '0.00', '0.00', '0.00', '0.000']],
   widths=[0.22, 0.13, 0.12, 0.12, 0.12, 0.12, 0.17],
   caption='Interaction energy minus the point-charge Coulomb energy (kcal/mol). The remaining value for the pairs with water is the charge-dipole interaction.')
p('The remaining value goes smoothly to 0 with distance. That the charges remain exact integers at 50 Å is the result of restricting charge transfer to nearby pairs.')

K.need(220)
h2('3.6 Spin-state energies')
tb([['System', 'DFT (ref.: restricted singlet)', 'Model', 'DFT (ref.: broken symmetry)', 'Model'],
    ['O<sub>2</sub>', '−34.7', '−32.5', '−12.4', '−10.7'],
    ['CH<sub>2</sub>', '−11.2', '−19.0', '−6.8', '−9.0'],
    ['NH', '−47.0', '−39.6', '−18.1', '−19.9'],
    ['SiH<sub>2</sub>', '+32.7', '+33.9', '+32.7', '+34.4'],
    ['[Fe(H<sub>2</sub>O)<sub>6</sub>]<super>2+</super>', '−76.6', '−40.9', '-', '−39.7']],
   widths=[0.24, 0.22, 0.14, 0.24, 0.16],
   caption='Energy difference between the high-spin state and the singlet, E(high spin) − E(singlet), kcal/mol.')
bl(['For small molecules the sign and magnitude are mostly right, and the difference between the restricted singlet and the broken-symmetry singlet (about 22 kcal/mol for O<sub>2</sub>) is also distinguished.',
    'The spin gap of the iron complex is at about half. There is little data for open-shell transition metals, and the per-element constants of the spin energy have initial values only for H, C, N and O.'])

K.need(470)
h2('3.7 Comparison with earlier runs')
fig(os.path.join(FIG, 'summary_vs_earlier_runs.png'),
    'Top: per-group energy and force errors of the internal validation and physics benchmark MAE compared with earlier runs. Bottom: ion-water scans at 0.05 Å spacing. '
    'The 4.5~5 Å step of the earlier runs (orange, grey dashed) is gone in LR-POLAR (blue).')

# =====================================================================================================================
h1('4. Limitations and next steps')
h2('4.1 Known problems')
tb([['Problem', 'Effect', 'How to fix'],
    ['max(0, ·) kink in the spin energy', 'Forces may be discontinuous', 'Replace with a smooth function'],
    ['Conductance exceeds the upper limit inside 2 Å', 'Charge transfer may be excessive at bond distances', 'Apply the decay before the upper limit'],
    ['Auxiliary loss weights proportional to the number of GPUs', 'Changing the number of GPUs changes the loss balance', 'Unify with a global normalisation'],
    ['Noble gases repel', 'Cannot be used for systems containing Ar, Xe', 'More data, or adjust the initial values of dispersion / repulsion'],
    ['Reference energy of isolated ions (LiF etc.)', 'Ionic-bond dissociation curves are off', 'Add isolated-ion and isolated-atom data'],
    ['Spin energy uses the 2S of the whole structure', 'Size consistency of the spin term is incomplete', 'Switch to local formal spins'],
    ['Automatic formal charges are wrong for about 10 %', 'Error on charged structures without labels', 'Extend the rules, re-evaluate with label formal charges']],
   widths=[0.34, 0.33, 0.33], caption='Items to fix before the next training.', align_right_from=-1)
h2('4.2 Plan for the next compute window')
bl(['<font name="NanumB">Data:</font> train on the full OMol25 (101,666,280 structures). The number of structures seen in this training (about 120 million) corresponds to 1.2 passes over the full data.',
    '<font name="NanumB">Several datasets:</font> use OPoly26 (polymers, same DFT level as OMol25), OC25 (surface catalysis, RPBE+D3) and MPtrj / sAlex (bulk) together. '
    'Where the DFT level differs, separate reference energies and output heads are used.',
    '<font name="NanumB">Comparison target:</font> the Thole-based model of BAM_v1 trained on the same data (4.5 million parameters, WBM formation-energy MAE about 33 meV/atom).'])

# ---- preprocessing status (read at build time)
def prep_row(name, label, level, d):
    try:
        m = json.load(open(os.path.join(d, 'meta.json')))
        s = json.load(open(os.path.join(d, 'sizes_train', 'sizes_meta.json')))
        return [label, level, f"{s['n']:,}", f"{s['avg_n_neighbors']:.1f}", f"{s['max_atoms']} / {s['max_edges']:,}",
                f"{m['e0_residual_std_eV_per_atom']:.3f}", 'done']
    except Exception:
        try:
            s = json.load(open(os.path.join(d, 'sizes_train', 'sizes_meta.json')))
            return [label, level, f"{s['n']:,}", f"{s['avg_n_neighbors']:.1f}", f"{s['max_atoms']} / {s['max_edges']:,}", '-',
                    'size table done']
        except Exception:
            return [label, level, '-', '-', '-', '-', 'in progress']

K.need(200)
h2('4.3 Preprocessing status')
rows = [['Dataset', 'DFT level', 'Structures (train)', 'Mean neighbours', 'Max atoms / edges', 'E0 residual (eV/atom)', 'Status']]
r4 = prep_row('omol4m', 'OMol25 4M', 'ωB97M-V', '/appsdata/t510a13/omol25/prep_4M_c6')
r4[5], r4[6] = '0.420', 'done (used for LR-POLAR training)'
rows.append(r4)
rows.append(prep_row('opoly26', 'OPoly26', 'ωB97M-V', '/appsdata/t510a13/prep/opoly26_c6.0'))
rows.append(prep_row('oc25', 'OC25', 'RPBE+D3', '/appsdata/t510a13/prep/oc25_c6.0'))
rows.append(prep_row('omol_full', 'OMol25 full', 'ωB97M-V', '/appsdata/t510a13/prep/omol_full_c6.0'))
tb(rows, widths=[0.15, 0.12, 0.16, 0.12, 0.17, 0.14, 0.14],
   caption='Preprocessing for stream training (size tables at a cutoff distance of 6 Å and per-element reference energies). The result summaries are in scripts/data/prep_results of the package.',
   font=8.2)

h1('Appendix. File locations')
tb([['Content', 'Location'],
    ['Training code', '/scratch/t510a13/BAM_V2_run_v3 (commit d90bdc5)'],
    ['Final checkpoint', '/appsdata/t510a13/race_runs/omol4m/omol4m_v3_polarqeq_c6_cs_ct_L5x128_r12p6_spec_hub_spinE_nbo_csloc_decL_disp_xrn3_lr1e-2_soap_g48x12_s42'],
    ['Training log, test results, figures', 'docs/results/race_v2_lr_polar_omol25_4m (repository BAM_V2, branch main)'],
    ['Preprocessing results (E0, meta)', 'scripts/data/prep_results'],
    ['Size tables', '/appsdata/t510a13/prep, /appsdata/t510a13/omol25/prep_4M_c6'],
    ['Plan for the next compute window', 'docs/PLAN_omol_full_next_window.md']],
   widths=[0.27, 0.73], align_right_from=-1)
note('Name: the name of this model is RACE v2 LR-POLAR. The names of the training run folder and the checkpoint folder still contain v3, the internal run name at training time '
     '(e.g. BAM_V2_run_v3). The Python package name is race_v2 (old name bam_omat24). v2local etc. are the names of earlier runs.')
note('The equations were transferred directly from the code used for training. The numbers in the tables are the results of the final full test of 2026-11-16 (Slurm jobs 1285135~1285139, 1285153).')

K.build(OUT, 'RACE v2 LR-POLAR: methods and OMol25 4M training results', footer='RACE v2 LR-POLAR methods and results · 2026-11-16')
print('wrote', OUT)
