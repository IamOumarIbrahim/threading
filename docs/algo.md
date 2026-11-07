**Threader algorithms — mathematical formalisation**

### Shared tone model
Given target colour $t \in [0,1]^3$ and background $bg \in [0,1]^3$, the **ink need** (minimal coverage) is
$$
q(t,bg) = \max_c\Bigl\{\,
\begin{cases}
\frac{t_c-bg_c}{1-bg_c} & t_c>bg_c,\\
\frac{bg_c-t_c}{bg_c} & t_c\le bg_c
\end{cases}
\Bigr\}
$$
clipped to $[0,1]$.  
The colour drawn by a thread is the most saturated colour on the ray from $bg$ through $t$:
$$
c(t,bg) = bg + \frac{t-bg}{\max\bigl(q(t,bg),0.05\bigr)}.
$$
Tone is realised solely by the local density of unit-width threads.

### 1. Greedy residual (analysis-by-synthesis)
Maintain a canvas $V$ initially equal to $bg$ and residual error
$$
E = \|T-V\|_2^2.
$$
At each step sample $K$ candidate segments whose mid-points are drawn from the residual mass, half of them aligned with the edge-tangent field.  
For a candidate $\ell$ and opacity $\alpha\in\{0.3,0.6,1\}$ the optimal colour is obtained in closed form and the error reduction
$$
\Delta E(\ell,\alpha) = E_{\text{old}} - E_{\text{new}}(\ell,\alpha)
$$
is maximised. The best candidate is locally polished by random endpoint nudges and committed; the process continues until a target thread count is reached.

### 2. Flow streamlines
Compute the structure-tensor edge-tangent field $(v_x,v_y)$.  
Partition the image into $4\times4$ cells and maintain a deficit
$$
D_{ij} = \sum_{(x,y)\in\text{cell}_{ij}} q(T_{xy},bg).
$$
Seeds are sampled from cells with large unpaid deficit. From each seed a streamline is integrated in both directions along $(v_x,v_y)$ until a sharp turn, the border, or an already-paid cell is encountered. The path is piecewise-linearised into long chords; each chord pays one unit of deficit for every cell it crosses.

### 3. Scanline dashes
Treat every row $y$ as a 1-D signal $q_y(x)$.  
Choose a mass $M$ (controlled by the budget) and cut the cumulative integral
$$
C_y(x) = \int_0^x q_y(s)\,ds
$$
into successive intervals of mass $M$. In each interval place a single horizontal dash of length $M$ centred at the centre-of-mass. Neighbouring dashes of similar colour are merged.

### 4. Radon peel
Build the discrete Radon table
$$
S(\theta,\rho) = \sum_{(x,y)\in\text{chord}(\theta,\rho)} \max\bigl(q(T_{xy},bg)-\tau,0\bigr)
$$
over 120 angles. Iteratively extract the strongest chord, retain only the contiguous runs where residual ink remains, split them by colour drift, and subtract a thin band of ink from the residual map (updating $S$ incrementally).

### 5. Stipple tour
Place a jittered point in every cell of a grid of spacing $g$ and keep it with probability proportional to a blurred, normalised ink need.  
Connect the retained points by a single open tour that always steps to the nearest unvisited point (greedy nearest-neighbour). Edges longer than a hop threshold are omitted (pen-up).

All five procedures therefore reduce the continuous tone-reproduction problem to the placement of a sparse set of 1-pixel-wide coloured line segments whose local density realises the required coverage $q$.

---

### Core primitives for algorithms 6–10
Image $I\in\{0,\dots,255\}^{H\times W\times 4}$ (RGBA) or normalised float buffer $T \in [0, 1]^{H \times W \times 3}$.  
Gradient field $(g_x,g_y)$ and magnitude $m=\sqrt{g_x^2+g_y^2}$ are obtained by central differences on the luminance channel.  
A candidate segment $\ell=(x_1,y_1,x_2,y_2)$ is coloured by the mean RGB along $n$ equidistant samples; its opacity is a decreasing function of the colour variance of those samples:
$$
\alpha = \operatorname{clip}\bigl(255 - 2.2\,\sigma_{\text{RGB}},\, 70,\, 255\bigr) \cdot \frac{\bar{A}}{255}.
$$

### 6. Edge-guided tangent tracing
Sample centres $(x_c,y_c)$ with probability proportional to edge strength
$$
p(x,y)\propto\bigl(m(x,y)+6\bigr)\cdot\mathbf{1}_{A(x,y)>0}.
$$
At each centre orient a segment perpendicular to the local gradient
$$
\theta=\operatorname{atan2}(g_y,g_x)+\tfrac{\pi}{2}+\varepsilon,\qquad\varepsilon\sim\mathcal{N}(0,0.12^2)
$$
and set half-length inversely proportional to normalised magnitude.  
The resulting $N$ fixed-length chords become the thread set.

### 7. Greedy error-reduction loom
Maintain an importance map $\iota\leftarrow 2.5\,m+15$ (modulated by alpha).  
At step $i$ tournament-select a seed of high $\iota$, evaluate eight directional rays of random length, and keep the ray whose colour discrepancy
$$
E=\|c_{\text{end}}-c_{\text{start}}\|_1+\|c_{\text{mid}}-c_{\text{start}}\|_1
$$
is minimal. Suppress $\iota$ along the chosen ray so that subsequent threads spread across the canvas.

### 8. Quadtree structural cross-hatching
Recursively partition the domain into a priority queue ordered by
$$
\text{priority}=-\operatorname{Var}(I_{\text{cell}})\cdot\operatorname{area}^{0.65}.
$$
Stop when a target number of leaves is reached.  
Inside each leaf of width $w$, height $h$ draw a controlled number of segments aligned with the mean edge tangent (and its orthogonal) of length $\approx0.48\sqrt{w^2+h^2}$. Thread counts are distributed proportionally to leaf variance $\times\sqrt{\text{area}}$.

### 9. Anchor-graph proximity web
Importance-sample $M\approx0.45N$ anchor points. Sort them by spatial strips and form candidate edges between anchors whose Euclidean distance lies in $[2,0.09\min(H,W)]$.  
Rank candidates by colour coherence (opacity) and retain the top $N$ edges; short local chords pad the set if fewer candidates exist.

### 10. Continuous greedy thread painter
Maintain target image $T \in [0,1]^{H \times W \times 3}$, continuous canvas $C$ initialized to background $bg$, and residual error field:
$$
E(x,y) = \|T(x,y) - C(x,y)\|_2^2.
$$
Every step proposes $N_{\text{cand}} = 160$ candidate line segments $\ell = (x_1, y_1, x_2, y_2)$ in subpixel float coordinates. Candidate mid-points $(m_x, m_y)$ are importance-sampled from the discrete cumulative distribution of $E(x,y)$ (with $15\%$ uniform exploration), directions follow the structure-tensor edge angle $\theta(m_x, m_y)$ with Gaussian perturbation:
$$
\theta = \operatorname{atan2}(v_y, v_x) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 0.12^2),
$$
and segment lengths follow a running log-normal length distribution $\ln \ell \sim \mathcal{N}(\mu_{\ln \ell}, 0.6^2)$.

Each candidate segment is evaluated across $K=48$ equidistant sample points $p_k$. For thread opacity $\alpha \in (0, 1]$, the optimal colour in closed form is:
$$
u_k = T(p_k) - (1 - \alpha) C(p_k), \qquad c^*(\ell) = \operatorname{clip}\left(\frac{1}{\alpha K} \sum_{k=1}^K u_k,\, 0,\, 1\right),
$$
yielding an analytical error reduction metric:
$$
\Delta E(\ell) = \Bigl(\sum_{k=1}^K \|T(p_k) - C(p_k)\|_2^2 - E_{\text{after}}(c^*)\Bigr) \cdot \frac{\max(\|\ell\|_2, 1)}{K}.
$$

The top $P=4$ candidate lines are refined through three successive multi-scale mutation rounds with scale factors $s \in \{1.0, 0.5, 0.25\}$. For each parent, $C_{\text{mut}} = 12$ mutant lines are generated by perturbing endpoints with Gaussian noise:
$$
\delta \sim \mathcal{N}\bigl(0,\, \sigma^2\bigr), \qquad \sigma = \max\bigl(1.5,\, 0.1\|\ell\|_2\bigr) \cdot s.
$$

The winning candidate is rasterised using an antialiased continuous distance kernel of width $w=2.0$ (half-width $h_w = 1.0$):
$$
m(p) = \alpha \cdot \operatorname{clip}\bigl(h_w + 0.5 - \operatorname{dist}(p, \ell),\, 0,\, 1\bigr).
$$
The exact optimal 8-bit quantized colour $c^*$ is computed over the local bounding box:
$$
c^* = \operatorname{clip}\left(\frac{\sum_{p} m(p) \bigl[T(p) - (1 - m(p)) C(p)\bigr]}{\sum_{p} m(p)^2},\, 0,\, 1\right),
$$
and committed to the canvas: $C(p) \leftarrow C(p)(1 - m(p)) + m(p) c^*$. Thread coordinates are stored as subpixel IEEE 754 half-precision floats (`float16`) in the $(N \times 8)$ thread matrix.


