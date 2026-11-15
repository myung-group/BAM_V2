"""RACE v3: methods (with equations) and results of the OMol25 4M run -- Korean PDF.

usage: python -I build_report.py <font_dir> <fig_dir> <out.pdf>
Equations follow the code the model was trained with (BAM_V2_run_v3, commit d90bdc5); see v3_methods_formulas.md.
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
K.title('RACE v3: 방법론과 OMol25 4M 학습 결과',
        '2026-11-16 · 모델 RACE_POLAR (파라미터 19,919,663개) · 학습 코드 BAM_V2_run_v3 (커밋 d90bdc5) · '
        'KISTI HANGANG, GPU 12개, 170 에폭, 21.1시간')

h1('요약')
p('RACE v3는 등변 메시지 전달 백본 위에 물리 항을 명시적으로 얹은 원자간 퍼텐셜이다. 전하는 전하 평형(Qeq)으로 구하되 '
  '분자별 형식전하에서 출발해 가까운 원자쌍 사이로만 흐르게 했고, 정전기·분극·분산·스핀 에너지를 각각 별도 항으로 계산한다. '
  '이 문서는 구현한 방법을 코드에 있는 수식 그대로 정리하고, OMol25 4M으로 170 에폭 학습한 결과를 싣는다.')
tb([['항목', '70 에폭', '최종 (170 에폭)', '비교'],
    ['내부 검증 6,000 구조 (E meV / F meV/Å)', '136.7 / 26.0', '90.9 / 24.2', '이전 버전 v2local: 188 / 24.8'],
    ['neutral_val 테스트 (E / F)', '-', '62.2 / 20.7', '에폭 16: 138.5 / 28.2'],
    ['공식 val 표본 32,000 구조 (E / F)', '448.7 / 24.8', '363.8 / 23.3', 'UMA-s-1.2.1: 44.9 / 7.9'],
    ['이온-물 곡선 MAE (kcal/mol)', '0.70~1.93', '0.33~1.04', 'UMA, eSEN: 2~3 (F<super>-</super>는 12~20)'],
    ['염다리 MAE (kcal/mol)', '0.74', '0.48', ''],
    ['크기 일관성 오차, 6 Å (kcal/mol)', '±0.5 이내', '±0.6 이내', '전체 전하를 입력하는 모델에서는 깨짐']],
   widths=[0.36, 0.17, 0.19, 0.28], caption='핵심 결과. E는 에너지 MAE, F는 힘 MAE다.', align_right_from=1)
bl(['<font name="NanumB">잘 되는 것:</font> 이온-물 상호작용 곡선은 비교한 공개 모델보다 정확하고, 떨어진 조각의 '
    '에너지가 고전 정전기로 정확히 수렴한다. 4.5 Å에서 보이던 에너지 계단은 사라졌다.',
    '<font name="NanumB">아직 부족한 것:</font> 공식 val의 평균 오차는 공개 모델의 약 8배다. 학습 데이터가 4M(전체의 4 %)이고, '
    '전하가 큰 구조와 금속·열린 껍질 구조에서 오차가 크다. 비활성 기체와 LiF 해리 곡선은 틀린다.',
    '<font name="NanumB">다음 단계:</font> OMol25 전체(1억 구조)와 OPoly26, OC25로 학습한다. 사전처리는 이번 계산 창에서 마쳤거나 진행 중이다.'])

# =====================================================================================================================
h1('1. 모델 개요')
p('총에너지는 원자별 에너지의 합이고, 힘은 총에너지를 좌표로 한 번 자동 미분해서 얻는다. 전하를 구하는 선형 방정식과 유도 '
  '쌍극자의 반복 풀이까지 미분에 포함되므로 힘은 에너지와 정확히 일관된다.')
eq(r'E=\sum_i\left[s\left(e^{\mathrm{rep}}_i+\sum_{t=0}^{4}\left(e^{\mathrm{node}}_{t,i}+e^{x}_{t,i}+e^{f}_{t,i}\right)\right)+b\right]'
   r'+\sum_i\left(e^{\mathrm{Qeq}}_i+e^{\mathrm{spin}}_i+e^{\mathrm{elec}}_i+e^{\mathrm{disp}}_i\right)+\sum_i E^{0}_{Z_i}')
eq(r'\mathbf{F}_i=-\dfrac{\partial E}{\partial \mathbf{r}_i}')
tb([['항', '내용', '범위', '절'],
    ['e<super>rep</super>', 'UFF 형태의 짧은 거리 쌍 반발', '6 Å 이웃', '2.1'],
    ['e<super>node</super>, e<super>x</super>, e<super>f</super>', '층마다 읽어 내는 단거리 원자 에너지 (5개 층)', '6 Å 이웃, 5회 전달', '2.1'],
    ['e<super>Qeq</super>', '전하 평형의 원자 자리 항 (전기음성도, 경도)', '구조 전체', '2.2~2.3'],
    ['e<super>spin</super>', '스핀 분극 에너지 (국소 모멘트와 쌍 상관)', '6 Å 이웃', '2.6'],
    ['e<super>elec</super>', '전하·쌍극자·사중극자의 쿨롱 에너지와 유도 분극', '구조 전체 (절단 없음)', '2.8'],
    ['e<super>disp</super>', 'C6 분산', '구조 전체 (절단 없음)', '2.9'],
    ['E<super>0</super>', '원소별 기준 에너지 (고정값)', '-', '2.10']],
   widths=[0.17, 0.47, 0.24, 0.12], caption='총에너지를 이루는 항. s와 b는 학습되는 스칼라(초기값 1과 0)다.',
   align_right_from=3)
p('단위는 Å, e, eV이고 쿨롱 상수는 K = 14.3996 eV·Å이다. 이 문서에서 S(r; a, b)는 r ≤ a에서 0, r ≥ b에서 1이 되는 매끄러운 '
  '스위치 함수다.')
eq(r'S(r;a,b)=x^{3}\left(10-15x+6x^{2}\right),\qquad x=\mathrm{clip}\left(\dfrac{r-a}{b-a},\,0,\,1\right)')

# =====================================================================================================================
h1('2. 방법론')
h2('2.1 단거리 백본')
p('원자를 노드로, 6 Å 안의 원자쌍을 간선으로 하는 그래프에서 등변 메시지를 5번 전달한다. 노드 특징은 l = 0~3의 구면 텐서 '
  '(종류마다 128채널)다. 거리 입력은 베셀 함수 12개와 원자 껍질 반지름으로 만든 지수 함수 16개이고, 다항식 절단 함수 f<sub>cut</sub>를 곱한다.')
eq(r'B_n(r)=\dfrac{2}{r_c}\,\dfrac{\sin(n\pi r/r_c)}{r}\ \ (n=1,\ldots,12),\qquad '
   r'f_{\mathrm{cut}}(r)=1-28u^{6}+48u^{7}-21u^{8},\ \ u=\dfrac{r}{r_c},\ \ r_c=6\ \mathrm{\AA}')
eq(r's_{ab}(r)=\exp\left(-\dfrac{r}{\rho_a(Z_i)+\rho_b(Z_j)}\right)f_{\mathrm{cut}}(r),\qquad a,b\in\{s,p,d,f\}')
p('층 t의 메시지 m, 원자 기저 A에서 만든 x, 둘의 텐서곱 p로 특징을 갱신한다. 평균 이웃 수(32.58)로 나누어 크기를 맞춘다.')
eq([r'A_i=\dfrac{1}{\bar{n}}\sum_{j\in N(i)}R^{(0)}(r_{ij})\odot\left[W^{(0)}_1 e_j\otimes Y(\hat{\mathbf{r}}_{ij})\right],\qquad '
    r'm_i=\dfrac{1}{\bar{n}}\sum_{j\in N(i)}R^{(t)}(r_{ij})\odot\left[W_1\tilde{h}_j\otimes Y(\hat{\mathbf{r}}_{ij})\right]',
    r'x_i=\hat{W}_t A_i,\qquad p_i=W_2\left[(W_m m_i)\otimes x_i\right],\qquad \delta_i=\gamma\odot\left(p_i+S_{Z_i}\tilde{h}_i\right)'])
eq(r'h^{(t+1)}=\delta\ \ (t=0,1),\qquad h^{(t+1)}=2\delta-h^{(t-1)}\ \ (t\geq 2)')
bl(['Y는 실수 구면 조화 함수, R<super>(t)</super>는 층마다 따로 학습하는 거리 MLP(28 → 64 → 64 → 64), e<sub>j</sub>는 원소 임베딩이다.',
    '<font name="NanumB">스펙트럼 정규화:</font> x를 만드는 가중치는 블록마다 가장 큰 특이값으로 나눈다(Ŵ). 멀리 있는 이웃의 약한 '
    '신호가 증폭되어 절단 거리에서 계단이 생기던 문제를 막기 위한 것이다.',
    '<font name="NanumB">체비셰프 잔차:</font> 세 번째 층부터 h<super>(t+1)</super> = 2δ − h<super>(t−1)</super> 형태의 점화식을 쓴다.',
    '원소별 가중치 표(S<sub>Z</sub> 등)는 랭크 32로 분해해 94개 원소가 파라미터를 공유한다.'])
p('각 층에서 세 가지 스칼라 에너지를 읽어 더한다. LN은 층 정규화이고, x 판독의 정규화 상수는 (3×10<super>−3</super>)<super>2</super>이다.')
eq(r'e^{\mathrm{node}}_{t,i}=w\cdot\mathrm{gate}\left(W_r h^{(t+1)}_i\right),\qquad '
   r'e^{x}_{t,i}=\mathrm{MLP}^{x}_{Z_i}\left(\dfrac{\mathrm{LN}(x_i)}{\sqrt{\bar{n}}}\right),\qquad '
   r'e^{f}_{t,i}=\mathrm{MLP}^{f}_{Z_i}\left(\dfrac{\mathrm{LN}(\tilde{h}^{(t)}_i)}{\sqrt{\bar{n}}}\right)')
p('아주 짧은 거리의 반발은 학습 데이터가 드물어서 UFF 표의 값으로 만든 쌍 퍼텐셜을 넣었다.')
eq(r'e^{\mathrm{rep}}_i=\dfrac{1}{2}\sum_{j\in N(i)}D_{ij}\,\dfrac{6}{\zeta_{ij}-6}\,'
   r'\exp\left[\zeta_{ij}\left(1-\dfrac{r_{ij}}{x_{ij}}\right)\right]f_{\mathrm{cut}}(r_{ij})\,'
   r'\mathrm{sigmoid}\left(\dfrac{r_{ij}-1.25\,(R_i+R_j)}{0.3\ \mathrm{\AA}}\right)')

h2('2.2 전하 평형 (Qeq)')
p('신경망이 원자마다 전기음성도 χ와 경도 J를 예측하고, 전하는 그 값으로 정의한 에너지 범함수를 최소화해서 얻는다. '
  '원자 전하는 폭 σ인 가우시안 분포로 본다.')
eq([r'\chi_i=\chi^{0}_{Z_i}+z_{i,1},\qquad J_i=\mathrm{softplus}\left(J^{0}_{Z_i}+z_{i,2}\right),\qquad '
    r'\sigma_i=r^{\mathrm{cov}}_{Z_i}\left[0.5+1.5\,\mathrm{sigmoid}(u_{Z_i})\right]',
    r'A_{ii}=J_i+\dfrac{K}{\sqrt{\pi}\,\sigma_i},\qquad '
    r'A_{ij}=K\,\dfrac{\mathrm{erf}\left(r_{ij}/\sqrt{2(\sigma_i^{2}+\sigma_j^{2})}\right)}{r_{ij}}'])
eq(r'E_{\mathrm{solve}}(\mathbf{q})=\sum_i\chi_i q_i+\dfrac{1}{2}\sum_{i,j}q_i A_{ij}q_j')
bl(['z<sub>i</sub>는 마지막 층의 스칼라 특징에서 MLP(128 → 32 → 2)로 얻는다. χ<super>0</super>는 폴링 전기음성도로 초기화한다.',
    '<font name="NanumB">폭의 범위 제한:</font> σ는 원소별로만 학습하고 공유 반지름의 0.5~2배로 묶었다. 이전 실험에서 폭이 제한 없이 '
    '커져 전하가 구조와 무관해지는 현상이 있었기 때문이다.',
    '행렬 A는 구조 안의 모든 원자쌍에 대해 절단 없이 만든다. 구조마다 352칸짜리 블록에 담아 한 번의 조밀 선형 풀이로 전하를 얻는다.',
    '에너지에는 이 범함수의 원자 자리 항만 넣는다. 쌍 쿨롱 에너지는 2.8절의 정전기 블록에서 한 번만 센다.'])
eq(r'e^{\mathrm{Qeq}}_i=\chi_i q_i+\dfrac{1}{2}\left(J_i+\dfrac{K}{\sqrt{\pi}\,\sigma_i}\right)q_i^{2}')

h2('2.3 전도도 기반 전하 이동')
p('보통의 Qeq는 구조 전체에서 전하를 자유롭게 재분배하므로, 멀리 떨어진 분자 사이에도 전하가 흐르고 해리 한계에서 분수 전하가 남는다. '
  'v3에서는 전하가 분자별 형식전하 q<super>0</super>에서 출발해 전도도 c<sub>ij</sub>가 있는 가까운 쌍을 따라서만 이동한다.')
eq([r'\mathbf{q}=\mathbf{q}^{0}+B\mathbf{p},\qquad \mathbf{p}=\mathrm{argmin}_{\mathbf{p}}\left[E_{\mathrm{solve}}(\mathbf{q}^{0}+B\mathbf{p})'
    r'+\dfrac{1}{2}\sum_{\mathrm{pairs}}\dfrac{p_{ij}^{2}}{c_{ij}}\right]',
    r'\Longleftrightarrow\quad (I+LA)\,\mathbf{q}=\mathbf{q}^{0}-L\boldsymbol{\chi},\qquad L=\mathrm{diag}\left(\sum_j c_{ij}\right)-c'])
p('p<sub>ij</sub>는 쌍 사이로 옮겨 간 전하이고, 옮기는 데 p<super>2</super>/2c의 벌점이 붙는다. 전도도는 원소별 파라미터 두 개(g, b)로 정한다.')
eq([r'c_{ij}=\exp\left[s_{ij}-\beta_{ij}\,(r_{ij}-r_0)\right]\left[1-S(r_{ij};3.0,4.5)\right],\qquad r_0=2\ \mathrm{\AA}',
    r's_{ij}=\ln C-\mathrm{softplus}\left(\ln C-g_{Z_i}-g_{Z_j}\right),\qquad '
    r'\beta_{ij}=\beta_{\min}+\mathrm{softplus}\left(b_{Z_i}+b_{Z_j}\right)'])
bl(['C = 100 e<super>2</super>/eV는 전도도의 상한, β<sub>min</sub> = 2.5 Å<super>−1</super>는 감쇠율의 하한이다. 시작값은 β = 4 Å<super>−1</super>이다.',
    '<font name="NanumB">보존되는 것:</font> L의 열 합이 0이므로, 4.5 Å보다 짧은 간선으로 이어진 원자 무리마다 q<super>0</super>의 합이 '
    '정확히 보존된다. 전도도가 없는 원자는 q<super>0</super>를 그대로 갖고, 전도도가 매우 크면 그 무리 안의 Qeq 전하로 수렴한다.',
    '<font name="NanumB">학습되는 감쇠:</font> 감쇠가 없을 때는 4.5 Å에서 전하 이동이 갑자기 켜져 이온-물 에너지에 계단이 생겼다. '
    '거리에 따라 지수로 줄이고 감쇠율을 학습하게 해서 없앴다(그림 4).',
    '행렬 I + LA는 대칭이 아니어서 조밀 LU로 풀고, 전하 보존이 깨지지 않도록 float64로 계산한다.'])
note('주의: 감쇠 인자를 상한 뒤에 곱하므로 r &lt; 2 Å에서는 전도도가 상한 C를 넘는다. 결과에 문제는 보이지 않았지만 다음 학습에서 고칠 항목이다.')

h2('2.4 형식전하와 형식스핀')
p('분자는 결합(거리가 공유 반지름 합의 1.2배보다 짧은 쌍)으로 이어진 원자 무리로 정의한다. 분자 m의 정수 전하를 정하고 그 분자의 '
  '원자에 고르게 나눈 것이 q<super>0</super>이다.')
eq(r'Q_m=\mathrm{round}\left(\sum_{i\in m}v_i\right),\qquad q^{0}_i=\dfrac{Q_{m(i)}}{N_{m(i)}},\qquad \sum_m Q_m=Q')
p('원자별 값 v는 아래 순서로 처음 있는 것을 쓴다. 분자 전하의 합이 전체 전하 Q와 다르면 나머지가 가장 큰 분자부터 ±1씩 고친다.')
tb([['순서', '출처', '쓰이는 경우'],
    ['1', '사용자가 준 원자별 형식전하', '추론, 특수한 경우'],
    ['2', 'NBO 전하, 없으면 Löwdin 전하 레이블', '학습 (OMol25 대부분의 행)'],
    ['3', '결합 규칙에 의한 자동 지정', '추론 전용 (레이블이 없을 때)'],
    ['4', '분자가 하나이거나 중성이면 Q 또는 0', '레이블이 없는 단순한 경우'],
    ['5', 'Q/N을 모든 원자에 균등 분배', '마지막 수단 (크기 일관성이 깨짐)']],
   widths=[0.1, 0.5, 0.4], caption='형식전하 q<super>0</super>를 정하는 순서.', align_right_from=-1)
p('형식스핀 s<super>0</super>도 같은 방식이다. 분자가 하나면 2S = 다중도 − 1을 쓰고, 여럿이면 Löwdin 스핀 레이블의 분자별 합을 '
  '전자 수의 홀짝에 맞게 반올림한다.')
h3('자동 지정 (추론 전용)')
p('레이블이 없는 구조에서는 결합 규칙으로 형식전하를 정한다. 순서는 다음과 같다.')
bl(['결합을 찾고, 금속은 결합을 끊어 고정 이온 전하 또는 기본 산화수를 준다.',
    '원자가와 결합 수를 비교해 오늄(+1)과 음이온을 정하고, 짧은 결합(공유 반지름 합의 0.97배 미만)에서 최대 매칭으로 π 결합을 놓는다.',
    'S·P·할로겐의 초원자가, 나이트로·피리디늄·구아니디늄 같은 오늄 규칙을 적용한다.',
    '남은 원자가는 고립 전자쌍 음이온, 금속 옆의 탄소 음이온, 라디칼로 처리한다.',
    '마지막으로 전체 전하와 다중도에 맞도록 애매한 자리(고립된 H, 할로겐화물, 금속)를 정확한 조합 탐색으로 고른다.'])
p('학습 샤드 표본에서 레이블로 구한 q<super>0</super>와 행 기준 90.1 %, 공식 val 표본에서 89.97 % 일치했다. 공식 val의 첫 200개에서 '
  '균등 분배 대신 자동 지정을 쓰면 에너지 MAE가 486.5에서 274.2 meV로 줄었다(70 에폭 기준).')

h2('2.5 국소 전하·스핀 입력과 크기 일관성')
p('전하나 스핀 상태가 다른 계를 한 모델로 다루려면 그 정보를 입력으로 넣어야 한다. 구조 전체의 전하 Q와 스핀 S를 모든 원자에 '
  '똑같이 넣는 방식(공개 모델 UMA도 같다)은 크기 일관성을 깨뜨린다. 멀리 떨어진 이온이 하나 추가되면 Q가 바뀌어 다른 분자의 '
  '에너지까지 달라지기 때문이다. v3는 원자마다 주변 6 Å 안의 형식전하와 형식스핀만 더해 넣는다.')
eq(r'\bar{Q}_i=q^{0}_i+\sum_{j\in N(i)}w(r_{ij})\,q^{0}_j,\qquad 2\bar{S}_i=s^{0}_i+\sum_{j\in N(i)}w(r_{ij})\,s^{0}_j,\qquad '
   r'w(r)=1-S(r;4.5,6.0)')
eq(r'h^{(0)}_i=e_i+\left[W_{c2}\,\mathrm{silu}\left(W_{c1}[\bar{Q}_i,2\bar{S}_i]+T_q(\bar{Q}_i)+T_s(2\bar{S}_i)\right)+b\right]\left(1+G_{Z_i,0}\right)')
bl(['T<sub>q</sub>, T<sub>s</sub>는 정수 전하·스핀마다 한 행을 갖는 학습 표이고 행 사이는 선형 보간한다. 같은 은닉값이 1~4번째 층의 '
    '스칼라 특징에도 층별 가중치와 원소별 게이트 G를 거쳐 더해진다.',
    '<font name="NanumB">크기 일관성:</font> 6 Å보다 멀리 있는 조각은 입력에 들어오지 않으므로, 조각 하나의 에너지는 다른 조각의 '
    '존재와 무관하다. 모든 원자가 4.5 Å 안에 있는 작은 분자에서는 전체 전하를 넣는 모델과 같아진다.',
    '관련 가중치는 모두 0으로 초기화해서 학습 초기에는 입력이 없는 모델과 같다.'])

h2('2.6 스핀 헤드와 스핀 에너지')
p('스핀 에너지는 원자의 국소 모멘트 μ ≥ 0과 이웃쌍의 상관 c ∈ [−1, 1]이라는 두 불변량으로 쓴다. μ는 짝짓지 않은 전자 수에 해당한다.')
eq(r'\mu_i=u_g\left[\mu^{\mathrm{raw}}_i+w_i\max\left(0,\ 2S-\sum_k\mu^{\mathrm{raw}}_k\right)\right],\qquad '
   r'\mu^{\mathrm{raw}}_i=\mathrm{softplus}(\cdot)')
eq(r'E_{\mathrm{spin}}=\sum_i\left[\Delta_i\mu_i+\dfrac{1}{2}W_{Z_i}\mu_i^{2}\right]+\sum_{i<j}J_{ij}\,\mu_i\mu_j\,\dfrac{c_{ij}+c_{ji}}{2}')
eq(r'J_{ij}=A_{Z_iZ_j}\exp\left[-(\kappa_{Z_i}+\kappa_{Z_j})\left(r_{ij}-r^{\mathrm{cov}}_i-r^{\mathrm{cov}}_j\right)\right]\left[1-S(r_{ij};4,6)\right]')
bl(['Δ(원자 자리 에너지)와 c는 신경망이, W(원소별 스핀 결합 상수), κ, A는 원소별 파라미터가 준다. W는 H, C, N, O에 대해 '
    'spin-polarised DFTB의 값(−1.95, −0.62, −0.71, −0.76 eV)으로 초기화했다.',
    '<font name="NanumB">합 조건:</font> μ의 합은 2S보다 작을 수 없다. 모자라면 소프트맥스 가중치 w로 나누어 채운다.',
    '<font name="NanumB">제한·비제한 구분:</font> 제한 계산으로 얻은 단일항은 u<sub>g</sub> = 0이어서 μ = 0, E<sub>spin</sub> = 0이다. '
    '열린 껍질과 비제한(깨진 대칭) 단일항은 u<sub>g</sub> = 1이다. 같은 구조의 두 해를 구분해 학습할 수 있다.',
    '보조 출력으로 부호 있는 스핀 밀도 m<sub>i</sub>도 예측한다. 합이 정확히 2S가 되도록 보정하며 에너지에는 들어가지 않는다.'])
eq(r'm_i=m^{\mathrm{raw}}_i+w_i\left(2S-\sum_k m^{\mathrm{raw}}_k\right)')
note('주의: max(0, ·)는 미분이 불연속인 꺾임을 만든다. 또 이 보정과 u<sub>g</sub>는 구조 전체의 값을 쓰므로 스핀 에너지는 2.5절의 '
     '국소 입력과 달리 완전히 크기 일관적이지는 않다.')

h2('2.7 NBO 재분배 헤드')
p('OMol25에는 Löwdin과 NBO 두 가지 전하·스핀 레이블이 있다. 정전기에 쓰는 전하 q는 Löwdin 전하에 맞추고, NBO 값은 이웃 사이의 '
  '재분배 사상으로 연결했다. 사상은 합을 보존한다.')
eq(r'R_t[v]_i=v_i+\sum_{j\in N(i)}t_{ij}\,(v_j-v_i),\qquad '
   r't_{ij}=t_{\max}\tanh\left(\dfrac{\mathrm{MLP}(\cdot)}{t_{\max}}\right)\left[1-S(r_{ij};3.0,4.5)\right],\quad t_{\max}=0.5')
bl(['로스에서는 사상을 레이블에 적용한다. 즉 R[q<super>Löwdin 레이블</super>]을 NBO 레이블과 비교하므로, 이 항은 쌍 계수 t만 학습시킨다.',
    'NBO 전하는 분자별 형식전하 q<super>0</super>를 정할 때 먼저 쓰이므로 모델에 간접적으로 들어간다.'])

h2('2.8 정전기와 분극')
p('원자마다 전하 q(2.3절), 영구 쌍극자 μ, 영구 사중극자 Θ, 분극률 α를 둔다. μ, Θ, α는 마지막 층 특징에서 등변 선형 사상으로 '
  '예측한다. 다중극 사이의 상호작용은 가우시안으로 뭉갠 쿨롱 핵을 쓴다.')
eq(r'T_a(r)=\dfrac{\mathrm{erf}(r/a)}{r},\qquad '
   r'\alpha_i=\alpha_{\min}+\left(\alpha^{\max}_{Z_i}-\alpha_{\min}\right)\mathrm{sigmoid}(\cdot)')
eq(r'a_0=2\sigma=2\ \mathrm{\AA}\ \ (\mathrm{permanent}),\qquad s_i=\alpha_i^{1/3},\qquad '
   r'a_i=\sqrt{2(s_i^{2}+\sigma^{2})},\qquad a_{ij}=\sqrt{2(s_i^{2}+s_j^{2})}')
p('영구 다중극이 만드는 장 F 아래에서 유도 쌍극자를 자기 일관적으로 푼다. 유도 쌍극자끼리의 상호작용 T도 포함한다.')
eq(r'\left(\dfrac{1}{\alpha_i}+\epsilon\right)\boldsymbol{\mu}^{\mathrm{ind}}_i-\left(T\boldsymbol{\mu}^{\mathrm{ind}}\right)_i=\mathbf{F}_i')
eq([r'e^{\mathrm{perm}}_i=\dfrac{K}{2}\left[q_i\phi_i-\boldsymbol{\mu}_i\cdot\mathbf{E}_i+\dfrac{1}{3}\Theta_i:G_i\right]',
    r'e^{\mathrm{pol}}_i=K\left[\dfrac{1}{2}\left(\dfrac{1}{\alpha_i}+\epsilon\right)\left|\boldsymbol{\mu}^{\mathrm{ind}}_i\right|^{2}'
    r'-\boldsymbol{\mu}^{\mathrm{ind}}_i\cdot\mathbf{F}_i-\dfrac{1}{2}\boldsymbol{\mu}^{\mathrm{ind}}_i\cdot\left(T\boldsymbol{\mu}^{\mathrm{ind}}\right)_i\right]',
    r'e^{\mathrm{elec}}_i=e^{\mathrm{perm}}_i+e^{\mathrm{pol}}_i'])
bl(['φ, <font name="NanumB">E</font>, G는 다른 모든 원자의 전하·쌍극자·사중극자가 원자 i에 만드는 퍼텐셜, 장, 장 기울기다. '
    '분자(열린 경계)에서는 구조 안의 모든 쌍을 절단 없이 더한다.',
    '유도 쌍극자는 켤레 기울기법으로 푼다(허용 오차 10<super>−4</super>, 최대 25회). 풀이 전체를 미분해 힘을 얻는다.',
    '분극률의 범위는 0.001~4 Å<super>3</super>(수소는 1 Å<super>3</super>까지)이고 시작값은 0.7 Å<super>3</super>이다. 유도 폭 s는 '
    '1 Å을 넘지 않게 매끄럽게 묶었다.',
    '구조의 쌍극자 모멘트는 Σ(q<sub>i</sub><font name="NanumB">r</font><sub>i</sub> + μ<sub>i</sub> + μ<sub>i</sub><super>ind</super>)이고 '
    'DFT 쌍극자 레이블에 맞춘다.'])
note('주의: 설정 이름은 thole_width이지만 실제 감쇠는 Thole의 지수 함수가 아니라 폭이 α<super>1/3</super>에 비례하는 가우시안 뭉개기다. '
     '또 전하의 자기 에너지가 Qeq 자리 항(원소별 폭)과 이 블록(폭 1 Å)에 한 번씩 들어 있다. 학습으로 흡수되는 상수 차이지만 정리가 필요하다.')

h2('2.9 분산')
p('C6 계수는 2.8절의 분극률에서 만든다. 따라서 분산 항도 분극률을 학습시키고, 분극률의 좌표 의존성이 힘에 들어간다.')
eq([r'C_{6,ii}=\mathrm{softplus}(\kappa_{Z_i})\,\alpha_i^{2},\qquad '
    r'C_{6,ij}=\dfrac{2\,C_{6,ii}\,C_{6,jj}}{\dfrac{\alpha_j}{\alpha_i}C_{6,ii}+\dfrac{\alpha_i}{\alpha_j}C_{6,jj}}',
    r'E_{\mathrm{disp}}=-\sum_{i<j}\dfrac{C_{6,ij}}{r_{ij}^{6}+\left(R^{0}_{ij}\right)^{6}},\qquad '
    r'R^{0}_{ij}=a_1\,(R_i+R_j),\qquad R_i=1.344\ \mathrm{\AA}\left(\dfrac{\alpha_i}{0.148\ \mathrm{\AA}^{3}}\right)^{1/7}'])
bl(['결합 규칙은 Tkatchenko-Scheffler, 감쇠는 Becke-Johnson 형태다. κ는 원소별로, a<sub>1</sub>은 하나의 스칼라로 학습한다(시작값 7.5 eV, 0.8).',
    '분자에서는 구조 안의 모든 쌍을 절단 없이 더한다.'])

h2('2.10 로스와 레이블 처리')
p('에너지 레이블은 DFT 에너지에서 원소별 기준 에너지의 합을 뺀 값이다. 에너지 잔차는 원자 수로 나눈다. 에너지와 힘에는 Huber 함수를 쓴다.')
eq(r'\mathcal{L}=10\,\mathcal{L}_E+\mathcal{L}_F+0.001\,\mathcal{L}_q+1.875\times10^{-5}\,\mathcal{L}_{\mathrm{dip}}+6.25\times10^{-4}\,\mathcal{L}_{\mathrm{spin}}')
eq([r'\mathcal{L}_E=\dfrac{1}{G}\sum_g H_{0.05}\left(\dfrac{E^{\mathrm{pred}}_g-E^{\mathrm{ref}}_g}{N_g}\right),\qquad '
    r'\mathcal{L}_F=\dfrac{1}{3N}\sum_i\sum_{a=x,y,z}H_{0.25}\left(F^{\mathrm{pred}}_{i,a}-F^{\mathrm{ref}}_{i,a}\right)',
    r'H_\delta(x)=\dfrac{x^{2}}{2}\ \ (|x|\leq\delta),\qquad H_\delta(x)=\delta\left(|x|-\dfrac{\delta}{2}\right)\ \ (|x|>\delta)'])
eq([r'\mathcal{L}_q=\dfrac{1}{N}\sum_i l_i\left(q_i-q^{\mathrm{L}}_i\right)^{2}+\left\langle\left(R_{t}[q^{\mathrm{L}}]_i-q^{\mathrm{NBO}}_i\right)^{2}\right\rangle,\qquad '
    r'\mathcal{L}_{\mathrm{dip}}=\left\langle\dfrac{1}{3}\left|\mathbf{p}_g-\mathbf{p}^{\mathrm{ref}}_g\right|^{2}\right\rangle',
    r'\mathcal{L}_{\mathrm{spin}}=\left\langle\left(\mu_i-|m^{\mathrm{ref}}_i|\right)^{2}\right\rangle+\left\langle\left(c_{ij}-\mathrm{sign}(m^{\mathrm{ref}}_i m^{\mathrm{ref}}_j)\right)^{2}\right\rangle'
    r'+\left\langle\left(m_i-m^{\mathrm{ref}}_i\right)^{2}\right\rangle+\left\langle\left(R_{t}[m^{\mathrm{ref}}]_i-m^{\mathrm{NBO}}_i\right)^{2}\right\rangle'])
bl(['Huber 경계는 에너지 0.05 eV/원자, 힘 0.25 eV/Å이다. Huber에 넣기 전에 잔차를 tanh로 한 번 더 눌러(에너지 5 eV/원자, 힘 10 eV/Å) '
    '극단적인 구조가 학습을 흔들지 못하게 했다.',
    '<font name="NanumB">없는 레이블:</font> 전하·스핀·쌍극자 레이블이 없는 행은 그 항의 가중치가 0이다. 에너지와 힘은 항상 학습한다.',
    '위첨자 L은 Löwdin 레이블, NBO는 NBO 레이블이고 p는 구조의 쌍극자 모멘트다.',
    '<font name="NanumB">스핀 상관:</font> c의 목표값은 두 원자의 스핀 레이블 부호의 곱이고, 두 원자 모두 |m| ≥ 0.1일 때만 센다.',
    '검증 로스는 10 L<sub>E</sub> + L<sub>F</sub>만으로 계산하며, 지수 이동 평균(EMA) 가중치로 평가해 가장 좋은 체크포인트를 고른다.'])
note('주의: ⟨·⟩로 쓴 항(NBO 전하, 쌍극자, 스핀)은 GPU마다 평균을 낸 뒤 더하므로 실제 가중치가 GPU 수(12)배가 된다. '
     '위 설정값 기준 0.012, 2.25×10<super>−4</super>, 7.5×10<super>−3</super>이다. GPU 수를 바꾸면 균형이 달라지므로 다음 학습에서 고칠 항목이다.')

K.need(230)
h2('2.11 학습 절차')
tb([['항목', '값'],
    ['데이터', 'OMol25 4M: 학습 3,936,920 구조(원자 2.16억 개), 검증 10,000 구조, neutral_val 테스트 5,000 구조'],
    ['구성', 'GPU 12개(3노드 × 4). 스텝당 약 556 그래프, 에폭당 1,306 스텝. 170 에폭은 데이터 약 32회 통과에 해당'],
    ['옵티마이저', 'SOAP (b1 = b2 = 0.95, 10 스텝마다 전처리 행렬 갱신). 0·1차원 파라미터와 원소별 표는 Adam'],
    ['학습률', '최대 0.01. 2 에폭 선형 워밍업 뒤 코사인으로 0.05배까지 감소'],
    ['그 밖', '기울기 노름 절단 1.0, EMA 0.999(초기 워밍업 포함), 모델은 float32이고 전하 풀이와 SOAP 통계는 float64'],
    ['비용', '에폭당 약 448초, 총 21.1시간 (Slurm 작업 1280748)']],
   widths=[0.17, 0.83], caption='학습 설정.', align_right_from=-1)
eq(r'\eta(t)=\eta_0\left[0.05+0.95\cdot\dfrac{1}{2}\left(1+\cos\dfrac{\pi\,(t-t_w)}{T-t_w}\right)\right],\qquad '
   r'd_t=\min\left(0.999,\ \dfrac{1+t}{10+t}\right)')
p('학습이 끝난 뒤 학습 데이터 표본으로 원자당 에너지 상수를 한 번 다시 맞춘다(게이지 보정). 이번에는 +0.145 meV/원자로 사실상 0이었다.')

# =====================================================================================================================
K.page()
h1('3. 결과')
h2('3.1 학습 곡선')
fig(os.path.join(FIG, 'v3_epochs_final.png'),
    '에폭별 로스와 검증 오차. 위 왼쪽: 전체 로스. 위 오른쪽: 학습 로스의 항별 값. 아래: 스트림 검증(모든 도메인)과 '
    'neutral_val 테스트의 에너지·힘 MAE.')
bl(['검증 로스는 143 에폭 무렵부터 0.00229~0.00230에서 멈췄고, 학습 로스만 계속 내려가 격차가 벌어졌다. 올라가지는 않았으므로 '
    '과적합으로 나빠지지는 않았지만, 4M 데이터로 얻을 수 있는 일반화는 거의 다 얻은 것으로 보인다.',
    '에너지는 끝까지 조금씩 좋아졌고(neutral_val 138.5 → 62.2 meV), 힘은 90 에폭 이후 거의 변하지 않았다(25 meV/Å).',
    '가장 좋은 체크포인트는 마지막 에폭(170, 검증 로스 0.002291)이다.'])

K.need(330)
h2('3.2 검증 오차')
tb([['그룹', '구조 수', '70 에폭 E', '최종 E', '70 에폭 F', '최종 F'],
    ['전체', '6,000', '136.7', '90.9', '26.0', '24.2'],
    ['생체 분자', '1,179', '127.0', '55.4', '12.9', '11.5'],
    ['전해질', '1,188', '197.1', '114.8', '21.2', '19.5'],
    ['금속 착물', '1,153', '247.3', '204.3', '38.9', '37.0'],
    ['반응성', '1,218', '74.5', '61.1', '54.4', '52.0'],
    ['중성 (q = 0)', '-', '93.7', '55.5', '21.0', '19.3'],
    ['q = ±1', '-', '160.2', '108.3', '25.4', '23.8'],
    ['|q| ≥ 2', '-', '261.1', '197.5', '39.2', '37.2'],
    ['닫힌 껍질', '-', '112.5', '64.1', '20.3', '18.5'],
    ['열린 껍질', '-', '259.6', '227.1', '51.4', '49.9'],
    ['금속 포함', '-', '262.6', '221.6', '44.2', '42.4'],
    ['금속 없음', '-', '107.6', '60.7', '20.4', '18.6']],
   widths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
   caption='내부 검증 6,000 구조의 그룹별 오차 (E: meV, F: meV/Å). 레이블에서 구한 형식전하를 쓴다.')
tb([['그룹', '구조 수', '70 에폭 E', '최종 E', '70 에폭 F', '최종 F'],
    ['전체', '32,000', '448.7', '363.8', '24.8', '23.3'],
    ['생체 분자', '7,013', '326.6', '206.7', '13.6', '12.3'],
    ['전해질', '19,725', '539.2', '452.7', '26.8', '25.2'],
    ['금속 착물', '2,242', '380.2', '330.9', '55.8', '54.5'],
    ['반응성', '2,598', '199.1', '183.4', '68.6', '66.4'],
    ['중성 (q = 0)', '13,572', '278.5', '192.5', '17.3', '16.0'],
    ['q = ±1', '13,844', '342.9', '262.1', '23.4', '21.8'],
    ['|q| ≥ 2', '4,584', '1272.2', '1178.1', '46.8', '45.2'],
    ['닫힌 껍질', '29,070', '420.7', '329.5', '21.4', '19.8'],
    ['열린 껍질', '2,930', '726.3', '703.7', '67.4', '66.7'],
    ['<font name="NanumB">UMA-s-1.2.1 (전체)</font>', '32,000', '', '44.9', '', '7.9'],
    ['<font name="NanumB">eSEN-md (전체)</font>', '32,000', '', '47.9', '', '4.8']],
   widths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
   caption='OMol25 공식 val 표본 32,000 구조의 오차 (E: meV, F: meV/Å). 형식전하는 결합 규칙으로 자동 지정했다. 공개 모델은 같은 표본에서 평가했다.')
bl(['공식 val은 내부 검증보다 훨씬 어렵다. 큰 전해질 클러스터가 60 %를 넘고 전하가 큰 구조가 많다.',
    '|q| ≥ 2 그룹의 오차(1,178 meV)가 전체 평균을 끌어올린다. 자동 지정한 형식전하가 틀린 행(약 10 %)이 주된 원인으로 의심되지만, '
    '레이블 형식전하로 다시 평가해 확인하지는 않았다.',
    '공개 모델과의 차이는 대부분 데이터 양에서 온다고 본다. UMA와 eSEN은 OMol25 전체(1억 구조)로 학습했고 v3는 4 %만 썼다. '
    '확인하려면 전체 데이터 학습이 필요하다.'])

K.need(300)
h2('3.3 물리 벤치마크')
tb([['곡선', '70 에폭', '최종', 'DFT 최소점', '최종 모델'],
    ['염다리 (아세테이트…메틸암모늄)', '0.74', '0.48', '−109.2', '−108.9'],
    ['NaCl 해리', '26.1', '9.5', '−98.1', '−108.7'],
    ['LiF 해리', '12.2', '20.0', '−137.3', '−161.0'],
    ['Li<super>+</super>…I<super>−</super>', '4.0', '2.9', '−136.9', '−137.0'],
    ['CH<sub>4</sub> 이량체', '0.05', '0.03', '−0.46', '−0.38'],
    ['벤젠 샌드위치', '0.36', '0.35', '−1.91', '−2.19'],
    ['할로겐 결합 (CF<sub>3</sub>Br…NH<sub>3</sub>)', '1.2', '1.2', '−3.74', '−1.84'],
    ['Ar<sub>2</sub>', '3.8', '3.0', '−0.28', '+3.66'],
    ['Li<super>+</super>…Xe', '31.0', '31.5', '−9.75', '+35.9']],
   widths=[0.36, 0.14, 0.14, 0.18, 0.18],
   caption='상호작용 에너지 곡선의 MAE와 DFT 최소점 위치에서의 값 (kcal/mol).')
fig(os.path.join(FIG, 'bench_vs_public_final.png'),
    '물리 벤치마크 곡선: DFT(검은 점), RACE v3 최종, 이전 버전(v2local), 공개 모델. 공개 모델은 6 Å 근처에서 상호작용이 끊겨 '
    '이온쌍 곡선이 평평해진다.', width=0.92)
bl(['이온쌍과 염다리의 먼 거리 꼬리(1/r)를 정확히 따라간다. 형식전하를 고정하고 정전기를 절단 없이 계산한 결과다.',
    'LiF는 학습 중간보다 나빠졌다. 최소점에서 24 kcal/mol 과하게 결합한다. 고립 이온의 에너지 기준이 맞지 않는 것으로 의심된다.',
    '비활성 기체(Ar<sub>2</sub>, Li<super>+</super>…Xe)는 인력이 있어야 할 거리에서 반발한다. 학습 데이터에 거의 없는 원소이고 '
    '학습이 진행돼도 좋아지지 않았다.',
    '할로겐 결합은 절반 정도만 재현한다.'])

K.need(260)
h2('3.4 이온-물 상호작용')
tb([['곡선', 'DFT 최소점', '모델 최소점', '최소점 오차', 'MAE (1.9~10 Å)', '10 Å에서 모델/DFT'],
    ['Li<super>+</super>', '−34.39', '−34.59', '−0.18', '0.41', '0.85'],
    ['Na<super>+</super>', '−24.25', '−24.71', '−0.46', '0.60', '0.85'],
    ['K<super>+</super>', '−17.52', '−17.49', '+0.03', '0.33', '0.85'],
    ['F<super>−</super> (C<sub>2v</sub>)', '−17.47', '−16.55', '+1.93', '1.04', '0.86'],
    ['F<super>−</super> (수소 결합)', '−24.41', '−23.27', '+1.13', '0.56', '0.88'],
    ['Cl<super>−</super> (C<sub>2v</sub>)', '−12.48', '−13.23', '−0.75', '0.67', '0.86'],
    ['Cl<super>−</super> (수소 결합)', '−13.89', '−14.01', '−0.09', '0.40', '0.87']],
   widths=[0.2, 0.15, 0.15, 0.15, 0.17, 0.18],
   caption='이온-물 상호작용 곡선 (kcal/mol). 기준은 ωB97M-V/def2-TZVPD 계산이다.')
fig(os.path.join(FIG, 'iw_vs_public_final.png'),
    '이온-물 상호작용 곡선: DFT와 RACE v3 최종, 이전 버전, 공개 모델의 비교. 오른쪽 아래는 곡선별 MAE다.')
bl(['v3의 MAE는 0.33~1.04 kcal/mol이다. 공개 모델은 2~3 kcal/mol이고, 6 Å 밖에서 상호작용이 0이 되며, F<super>−</super>에서는 '
    '12~20 kcal/mol까지 틀린다.',
    '이전 버전의 4.5~5 Å 계단은 없어졌다. 0.05 Å 간격 스캔에서 가장 큰 튐은 0.29~0.60 kcal/mol이고 모두 3.50~3.55 Å에서 나온다. '
    'K<super>+</super>와 F<super>−</super> 곡선에 작은 꺾임이 남아 있다.',
    '10~20 Å에서 상호작용이 DFT의 0.84~0.88배로 약 15 % 약하다. 물 분자의 쌍극자가 작게 예측되는 것으로 추정한다.'])

K.need(300)
h2('3.5 크기 일관성')
p('두 조각을 거리 R만큼 떼어 놓고, 모델의 상호작용 에너지에서 점전하 쿨롱 에너지를 뺀 값을 본다. 0에 가까울수록 좋다.')
tb([['쌍', 'R = 6 Å', '8 Å', '10 Å', '15 Å', '20 Å', '50 Å에서 전하'],
    ['Li<super>+</super>…I<super>−</super>', '−0.60', '−0.19', '−0.08', '−0.02', '0.00', '−1.000'],
    ['Na<super>+</super>…Cl<super>−</super>', '−0.59', '−0.18', '−0.07', '−0.01', '0.00', '−1.000'],
    ['Li<super>+</super>…Xe', '−0.40', '−0.13', '−0.05', '−0.01', '0.00', '0.000'],
    ['Na<super>+</super>…H<sub>2</sub>O', '+0.29', '+0.23', '+0.17', '+0.08', '+0.05', '0.000'],
    ['Cl<super>−</super>…H<sub>2</sub>O', '−0.25', '+0.18', '+0.13', '+0.07', '+0.04', '0.000'],
    ['F<super>−</super>…H<sub>2</sub>O', '−0.33', '+0.18', '+0.13', '+0.07', '+0.04', '0.000'],
    ['Ar…Ar', '−0.02', '0.00', '0.00', '0.00', '0.00', '0.000']],
   widths=[0.22, 0.13, 0.12, 0.12, 0.12, 0.12, 0.17],
   caption='상호작용 에너지에서 점전하 쿨롱 에너지를 뺀 값 (kcal/mol). 물이 있는 쌍의 남은 값은 전하-쌍극자 상호작용이다.')
p('남은 값은 거리에 따라 매끄럽게 0으로 간다. 50 Å에서 전하가 정확히 정수로 남는 것은 전하 이동을 가까운 쌍으로 제한한 결과다.')

K.need(220)
h2('3.6 스핀 상태 에너지')
tb([['계', 'DFT (제한 단일항 기준)', '모델', 'DFT (깨진 대칭 기준)', '모델'],
    ['O<sub>2</sub>', '−34.7', '−32.5', '−12.4', '−10.7'],
    ['CH<sub>2</sub>', '−11.2', '−19.0', '−6.8', '−9.0'],
    ['NH', '−47.0', '−39.6', '−18.1', '−19.9'],
    ['SiH<sub>2</sub>', '+32.7', '+33.9', '+32.7', '+34.4'],
    ['[Fe(H<sub>2</sub>O)<sub>6</sub>]<super>2+</super>', '−76.6', '−40.9', '-', '−39.7']],
   widths=[0.24, 0.22, 0.14, 0.24, 0.16],
   caption='높은 스핀 상태와 단일항의 에너지 차이 E(높은 스핀) − E(단일항), kcal/mol.')
bl(['작은 분자에서는 부호와 크기를 대체로 맞추고, 제한 단일항과 깨진 대칭 단일항의 차이(O<sub>2</sub>에서 약 22 kcal/mol)도 구분한다.',
    '철 착물의 스핀 갭은 절반 수준이다. 전이 금속 열린 껍질은 데이터가 적고, 스핀 에너지의 원소별 상수가 H, C, N, O에만 초기값이 있다.'])

K.need(470)
h2('3.7 이전 버전과의 비교')
fig(os.path.join(FIG, 'v3_summary_final.png'),
    '위: 내부 검증의 그룹별 에너지·힘 오차와 물리 벤치마크 MAE를 이전 버전과 비교. 아래: 이온-물 0.05 Å 간격 스캔. '
    '이전 버전(주황, 회색 점선)의 4.5~5 Å 계단이 v3(파랑)에서 없어졌다.')

# =====================================================================================================================
h1('4. 한계와 다음 단계')
h2('4.1 알려진 문제')
tb([['문제', '영향', '고칠 방법'],
    ['스핀 에너지의 max(0, ·) 꺾임', '힘이 불연속일 수 있음', '매끄러운 함수로 바꿈'],
    ['전도도가 2 Å 안쪽에서 상한을 넘음', '결합 거리에서 전하 이동이 과할 수 있음', '감쇠를 상한 앞에 적용'],
    ['보조 로스 가중치가 GPU 수에 비례', 'GPU 수를 바꾸면 로스 균형이 달라짐', '전역 정규화로 통일'],
    ['비활성 기체가 반발함', 'Ar, Xe가 든 계에 쓸 수 없음', '데이터 보강 또는 분산·반발 초기값 조정'],
    ['LiF 등 고립 이온의 기준 에너지', '이온 결합 해리 곡선이 어긋남', '고립 이온·원자 데이터 추가'],
    ['스핀 에너지가 구조 전체의 2S를 씀', '스핀 항의 크기 일관성이 불완전', '국소 형식스핀으로 바꿈'],
    ['자동 형식전하가 약 10 % 틀림', '레이블 없는 대전 구조의 오차', '규칙 보강, 레이블 형식전하로 재평가']],
   widths=[0.34, 0.33, 0.33], caption='다음 학습 전에 고칠 항목.', align_right_from=-1)
h2('4.2 다음 계산 창 계획')
bl(['<font name="NanumB">데이터:</font> OMol25 전체(101,666,280 구조)로 학습한다. 이번 학습이 본 구조 수(약 1.2억)는 전체 데이터 1.2회 통과에 해당한다.',
    '<font name="NanumB">여러 데이터셋:</font> OPoly26(고분자, OMol25와 같은 DFT 수준)과 OC25(표면 촉매, RPBE+D3), MPtrj·sAlex(벌크)를 함께 쓴다. '
    'DFT 수준이 다르면 기준 에너지와 출력 헤드를 따로 둔다.',
    '<font name="NanumB">비교 대상:</font> 같은 데이터로 학습한 BAM_v1의 Thole 기반 모델(파라미터 450만 개, WBM 형성 에너지 MAE 약 33 meV/원자).'])

# ---- preprocessing status (read at build time)
def prep_row(name, label, level, d):
    try:
        m = json.load(open(os.path.join(d, 'meta.json')))
        s = json.load(open(os.path.join(d, 'sizes_train', 'sizes_meta.json')))
        return [label, level, f"{s['n']:,}", f"{s['avg_n_neighbors']:.1f}", f"{s['max_atoms']} / {s['max_edges']:,}",
                f"{m['e0_residual_std_eV_per_atom']:.3f}", '완료']
    except Exception:
        try:
            s = json.load(open(os.path.join(d, 'sizes_train', 'sizes_meta.json')))
            return [label, level, f"{s['n']:,}", f"{s['avg_n_neighbors']:.1f}", f"{s['max_atoms']} / {s['max_edges']:,}", '-',
                    '크기표 완료']
        except Exception:
            return [label, level, '-', '-', '-', '-', '진행 중']

K.need(200)
h2('4.3 사전처리 현황')
rows = [['데이터셋', 'DFT 수준', '구조 수 (train)', '평균 이웃 수', '최대 원자 / 간선', 'E0 잔차 (eV/원자)', '상태']]
r4 = prep_row('omol4m', 'OMol25 4M', 'ωB97M-V', '/appsdata/t510a13/omol25/prep_4M_c6')
r4[5], r4[6] = '0.420', '완료 (v3 학습에 사용)'
rows.append(r4)
rows.append(prep_row('opoly26', 'OPoly26', 'ωB97M-V', '/appsdata/t510a13/prep/opoly26_c6.0'))
rows.append(prep_row('oc25', 'OC25', 'RPBE+D3', '/appsdata/t510a13/prep/oc25_c6.0'))
rows.append(prep_row('omol_full', 'OMol25 전체', 'ωB97M-V', '/appsdata/t510a13/prep/omol_full_c6.0'))
tb(rows, widths=[0.15, 0.12, 0.16, 0.12, 0.17, 0.14, 0.14],
   caption='스트림 학습용 사전처리(절단 거리 6 Å의 크기표와 원소별 기준 에너지). 결과 요약은 패키지의 data_prep/prep_results에 있다.',
   font=8.2)

h1('부록. 파일 위치')
tb([['내용', '위치'],
    ['학습 코드', '/scratch/t510a13/BAM_V2_run_v3 (커밋 d90bdc5)'],
    ['최종 체크포인트', '/appsdata/t510a13/race_runs/omol4m/omol4m_v3_polarqeq_c6_cs_ct_L5x128_r12p6_spec_hub_spinE_nbo_csloc_decL_disp_xrn3_lr1e-2_soap_g48x12_s42'],
    ['학습 로그, 테스트 결과, 그림', 'BAM_V2_wt_thole/docs/results_omol4m_v3'],
    ['사전처리 결과 (E0, 메타)', 'BAM_V2_wt_thole/data_prep/prep_results'],
    ['크기표', '/appsdata/t510a13/prep, /appsdata/t510a13/omol25/prep_4M_c6'],
    ['다음 계산 창 계획', 'BAM_V2_wt_thole/docs/PLAN_omol_full_next_window.md']],
   widths=[0.27, 0.73], align_right_from=-1)
note('수식은 학습에 쓴 코드에서 직접 옮겼다. 표의 숫자는 2026-11-16의 최종 총 테스트(Slurm 작업 1285135~1285139, 1285153) 결과다.')

K.build(OUT, 'RACE v3: 방법론과 OMol25 4M 학습 결과', footer='RACE v3 방법론과 결과 · 2026-11-16')
print('wrote', OUT)
