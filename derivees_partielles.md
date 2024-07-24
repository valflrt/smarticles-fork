Pour $(n, m) \in \mathbb N^2$, notons $\Theta_{n,m} = \mathcal M_{m,n}(\mathbb R) \times \mathbb R^m$.

On pose $a: x \in \mathbb R \longmapsto \max(x, 0)$

Sa dérivée est: $a': x \in \mathbb R^* \longmapsto \begin{cases} \; 1 & \text{si} \; x > 0 \\ \; 0 & \text{sinon} \end{cases}$

Pour un vecteur $X$, on note $\tilde a(X)$, le vecteur dont les coordonnées sont les images des coordonnées de $X$ par $a$

# Calcul des dérivées partielles

Soit $(m, n) \in \mathbb N^2$.

On pose $\varphi: (X, W, B) \in \mathbb R^m \times \Theta_{m,n} \longmapsto \tilde a(\underbrace{WX+B}_{\displaystyle Z}) \in \mathbb R^n$.

Notons $\varphi_1,\dots,\varphi_n$ les composantes de $\varphi$.

Soit $i \in \{1,\dots,n\}$. Pour tout $(X, W, B) \in \mathbb R^m \times \Theta_{m,n}$:

$$\begin{align*} \varphi_i (X, W, B) & = a([WX]_i + [B]_i) \\ & = a\Bigg(\underbrace{ \sum\limits_{k=1}^n [W]_{i,k} [X]_k + [B]_i}_{\displaystyle [Z]_i}\Bigg) \end{align*}$$

## Dérivées partielles des composantes de $\varphi$ par rapport aux coefficients de $X$

Fixons $W$ et $B$.

Soit $(i, j) \in \{1,\dots,n\} \times \{1,\dots,m\}$.

La dérivée partielle de $\varphi_i$ par rapport au coefficient $[X]_j$ est:

$$\dfrac{\partial \varphi_i}{\partial [X]_j} = [W]_{i,j} \times a'([Z]_i)$$

On somme ensuite les "contributions" pour obtenir quoi ? je ne sais pas:

$$\sum\limits_{i=1}^n [W]_{i,j} \times a'([Z]_i)$$

## Dérivées partielles des composantes de $\varphi$ par rapport aux coefficients de $W$

Fixons $X$ et $B$.

Soit $(i, j, p) \in \{1,\dots,n\} \times \{1,\dots,n\} \times \{1,\dots,m\}$.

La dérivée partielle de $\varphi_i$ par rapport au coefficient $[W]_{j,p}$ est:

$$\dfrac{\partial \varphi_i}{\partial [W]_{j,p}} = \begin{cases} \; [X]_p \times a'([Z]_j) \quad & \text{si} \; i = j \\ \quad\quad\quad 0 & \text{sinon} \end{cases}$$

Donc seul le cas $i=j$ nécessite d'être calculé.

## Dérivées partielles des composantes de $\varphi$ par rapport au coefficients de $B$

Fixons $X$ et $W$.

Soit $(i, j) \in \{1,\dots,n\} \times \{1,\dots,n\}$.

La dérivée partielle de $\varphi_i$ par rapport au coefficient $[B]_j$ est:

$$\dfrac{\partial \varphi_i}{\partial [B]_j} = \begin{cases} \; a'([Z]_j) \quad & \text{si} \; i = j \\ \quad\;\; 0 & \text{sinon} \end{cases}$$

Donc seul le cas $i=j$ nécessite d'être calculé.

## Dérivées partielles de la fonction coût $L$ par rapport aux coefficients de $Y$

Pour tout $j \in \{1,\dots,n\}$, la dérivée partielle de $L$ par rapport à $[Y]_j$ est:

$$\dfrac{\partial L}{\partial [Y]_i} = \sum\limits_{k=1}^n ([Y]_k-[Y]_k)$$

# Dérivées partielles de la composée de $L$ avec une fonction couche $\varphi$

Soit $(\ell_0, \ell_1, \ell_2, \ell_3) \in \mathbb N^4$.

La fonction coût $L$ est définie par:

$$L: (Y, \tilde Y) \in (\mathbb R^n)^2 \; \longmapsto \dfrac{1}{2} \|Y - \tilde Y\|^2$$

Pour tout $(m,n) \in \mathbb N^2$, on pose $\varphi^{(m,n)}: (X, W, B) \in \mathbb R^m \times \Theta_{m,n} \longmapsto \tilde a(\underbrace{WX+B}_{\displaystyle Z}) \in \mathbb R^n$ et on note ses composantes $\varphi^{(m,n)}_1,\dots,\varphi^{(m,n)}_n$.

Soit $(m,n) \in \mathbb N^2$.

Soit $(X, W, B) \in \mathbb R^m \times \Theta_{m,n}$. Pour tout $i \in \{1,\dots,n\}$, on a:

$$\begin{align*} \varphi^{(m,n)}_i (X, W, B) & = a([WX]_i + [B]_i) \\ & = a\Bigg(\underbrace{\sum\limits_{k=1}^n [W]_{i,k} [X]_k + [B]_i}_{\displaystyle [Z]_i}\Bigg) \end{align*}$$

Soit $\tilde Y \in \mathbb R^q$ et $(W^{(0)}, B^{(0)}, W^{(1)}, B^{(1)}, W^{(2)}, B^{(2)}) \in \Theta_{\ell_0,\ell_1} \times \Theta_{\ell_1,\ell_2} \times \Theta_{\ell_2,\ell_3}$.

- Pour tout $X^{(2)} \in \mathbb R^{\ell_2}$, on a:

  $$L(\varphi^{(\ell_2,\ell_3)}(X^{(2)}, W^{(2)}, B^{(2)}), \tilde Y) = \dfrac{1}{2} \sum\limits_{k=1}^{\ell_3} (\varphi^{(\ell_2,\ell_3)}_k(X^{(2)}, W^{(2)}, B^{(2)}) - [\tilde Y]_k)^2$$

- Pour tout $X^{(1)} \in \mathbb R^{\ell_1}$, on a:

  $$\begin{align*} & L(\varphi^{(\ell_1,\ell_2)}(\varphi^{(\ell_1,\ell_2)}(X^{(1)}, W^{(1)}, B^{(1)}), W^{(1)}, B^{(1)}), \tilde Y) \\ = \; & \dfrac{1}{2} \sum\limits_{k=1}^{\ell_2} (\varphi^{(\ell_2,\ell_3)}_k(\varphi^{(\ell_1,\ell_2)}(X^{(1)}, W^{(1)}, B^{(1)}), W^{(2)}, B^{(2)}) - [\tilde Y]_k)^2 \\ = \; & \dfrac{1}{2} \sum\limits_{k=1}^{\ell_2} \Bigg(\varphi^{(\ell_2,\ell_3)}_k\Bigg(\sum\limits_{k'=1}^{\ell_2} [W^{(2)}]_{k,k'} \varphi^{(\ell_1,\ell_2)}_{k'}(X^{(1)}, W^{(1)}, B^{(1)}), + [B^{(2)}]_k\Bigg) - [\tilde Y]_k\Bigg)^2 \end{align*}$$

- Pour tout $X^{(0)} \in \mathbb R^{\ell_0}$, on a:

  $$L(\varphi^{(\ell_0,\ell_1)}(X^{(0)}, W^{(0)}, B^{(0)}), \tilde Y) = \dfrac{1}{2} \sum\limits_{k=1}^{\ell_1} (\varphi^{(\ell_0,\ell_1)}_k(X^{(0)}, W^{(0)}, B^{(0)}) - [\tilde Y]_k)^2$$

Pour la suite, on fixe $\tilde Y$.

## Dérivées partielles de la fonction coût $L$ par rapport aux coefficients de $X$

Fixons $W$, $B$ et $\tilde Y$.

Pour tout $i \in \{1,\dots,m\}$, la dérivée partielle de $f: X \longmapsto L(\varphi_k(X, W, B), \tilde Y)$ par rapport à $[X]_i$ est:

$$\begin{align*} \dfrac{\partial f}{\partial [X]_i} & = \sum\limits_{k=1}^n \dfrac{\partial \varphi_k}{\partial [X]_i} \times (\varphi_k(X, W, B)-[\tilde Y]_k) \\ & = \sum\limits_{k=1}^n [W]_{k,i} \times a'([Z]_k) \times (\varphi_k(X, W, B)-[\tilde Y]_k) \end{align*}$$

## Dérivée partielle de la fonction coût $L$ par rapport aux coefficients de $W$

Fixons $X$, $B$ et $\tilde Y$.

Soit $(i,j) \in \{1,\dots,n\} \times \{1,\dots,m\}$. La dérivée partielle de $f: W \longmapsto L(\varphi(X, W, B), \tilde Y)$ par rapport à $[W]_{i,j}$ est:

$$\begin{align*} \dfrac{\partial f}{\partial [W]_{i,j}} & = \sum\limits_{k=1}^n \underbrace{\dfrac{\partial \varphi_k}{\partial [W]_{i,j}}}_{= \; 0 \; \text{si} \; i \neq k} \times (\varphi_k(X, W, B)-[\tilde Y]_k) \\ & = \dfrac{\partial \varphi_i}{\partial [X]_{i,j}} \times (\varphi_i(X, W, B)-[\tilde Y]_i) \\ & = [X]_j \times a'([Z]_i) \times (\varphi_i(X, W, B)-[\tilde Y]_i) \end{align*}$$

Fixons $W^{(2)}$, $B^{(2)}$, $X^{(1)}$, $B^{(1)}$ et $\tilde Y$.

Soit $(i,j) \in \{1,\dots,\ell_2\} \times \{1,\dots,\ell_1\}$. La dérivée partielle de:

$$\begin{align*} f: W^{(1)} \longmapsto \; & L(\varphi^{(\ell_2,\ell_3)}(\varphi^{(\ell_1,\ell_2)}(X^{(1)}, W^{(1)}, B^{(1)}), W^{(2)}, B^{(2)}), \tilde Y) \\ & =  \end{align*}$$

par rapport à $[W]_{i,j}$ est:

$$\begin{align*} \dfrac{\partial f}{\partial [W]_{i,j}} =  \end{align*}$$

## Dérivée partielle de la fonction coût $L$ par rapport aux coefficients de $B$

Fixons $X$, $W$ et $\tilde Y$.

Soit $j \in \{1,\dots,m\}$. La dérivée partielle de $f: B \longmapsto L(\varphi_k(X, W, B), \tilde Y)$ par rapport à $[B]_j$ est:

$$\begin{align*} \dfrac{\partial f}{\partial [B]_j} & = \sum\limits_{k=1}^n \underbrace{\dfrac{\partial \varphi_k}{\partial [B]_j}}_{= \; 0 \; \text{si} \; i \neq k} \times (\varphi_k(X, W, B)-[\tilde Y]_k) \\ & = \dfrac{\partial \varphi_j}{\partial [B]_j} \times (\varphi_j(X, W, B)-[\tilde Y]_j) \\ & = a'([Z]_j) \times (\varphi_j(X, W, B)-[\tilde Y]_j) \end{align*}$$

# Recherche d'une règle de dérivation en chaîne

Soit $(m,n,q) \in \mathbb N^3$.

Considérons les applications:

$$\varphi: (X, W, B) \in \mathbb R^m \times \Theta_{m,n} \longmapsto \tilde a(\underbrace{WX+B}_{\displaystyle Z}) \in \mathbb R^n$$

et:

$$\varphi': (X', W', B') \in \mathbb R^n \times \Theta_{n,q} \longmapsto \tilde a(\underbrace{W'X'+B'}_{\displaystyle Z'}) \in \mathbb R^q$$

Notons $(\varphi_1,\dots,\varphi_n)$ et $(\varphi'_1,\dots,\varphi'_q)$ les composantes de $\varphi$ et de $\varphi'$ respectivement.

Posons $\psi: (X, W, B, W', B') \in \mathbb R^m \times \Theta_{m,n} \times \Theta_{n,q} \longmapsto \varphi'(\varphi(X,W,B),W',B')$ et notons $\psi_1,\dots,\psi_q$ ses composantes.

Soit $i \in \{1,\dots,q\}$. Pour tout $(X, W, B, W', B') \in \mathbb R^m \times \Theta_{m,n} \times \Theta_{n,q}$:

$$\begin{align*} \psi_i(X, W, B, W', B') & = \varphi'_i(\varphi(X, W, B), W', B') \\ & = a\Bigg(\sum\limits_{k=1}^n [W']_{i,k} \varphi_k(X, W, B) + [B']_i\Bigg) \\ & = a\Bigg(\underbrace{\sum\limits_{k=1}^n [W']_{i,k} a\Bigg(\underbrace{\sum\limits_{k'=1}^n [W]_{k,k'} [X]_{k'} + [B]_k}_{\displaystyle [Z]_k}\Bigg) + [B']_i}_{\displaystyle [Z']_i}\Bigg) \end{align*}$$

## Dérivées partielles des composantes de $\psi$ par rapport aux coefficients de $X$

Fixons $W$, $B$, $W'$ et $B'$.

Soit $(i, j) \in \{1,\dots,q\} \times \{1,\dots,m\}$.

La dérivée partielle de $\psi_i$ par rapport au coefficient $[X]_j$ est:

$$\begin{align*} \dfrac{\partial \psi_i}{\partial [X]_j} & = \Bigg( \sum\limits_{k=1}^n \underbrace{[W]_{k,j} \times a'([Z]_k)}_{\dfrac{\partial \varphi_k}{\partial [X]_j}} \Bigg) \times a'([Z']_i) \\ & = a'([Z']_i) \times \sum\limits_{k=1}^n \dfrac{\partial \varphi_k}{\partial [X]_j} \end{align*}$$

## Dérivées partielles des composantes de $\psi$ par rapport aux coefficients de $W$

Fixons $X$, $B$, $W'$ et $B'$.

Soit $(i, j, p) \in \{1,\dots,q\} \times \{1,\dots,n\} \times \{1,\dots,m\}$.

La dérivée partielle de $\psi_i$ par rapport au coefficient $[W]_{j,p}$ est:

$$\begin{align*} \dfrac{\partial \psi_i}{\partial [W]_{j,p}} & = \Big( [W']_{i,j} \times \underbrace{[X]_p \times a'([Z]_j)}_{\dfrac{\partial \varphi_j}{\partial [W]_{j,p}}} \Big) \times a'([Z']_i) \\ & = [W']_{i,j} \times a'([Z']_i) \times \dfrac{\partial \varphi_j}{\partial [W]_{j,p}} \end{align*}$$

## Dérivées partielles des composantes de $\varphi$ par rapport au coefficients de $B$

Fixons $X$ et $W$.

Soit $(i, j) \in \{1,\dots,q\} \times \{1,\dots,n\}$.

La dérivée partielle de $\varphi_i$ par rapport au coefficient $[B]_j$ est:

$$\begin{align*} \dfrac{\partial \psi_i}{\partial [B]_j} & = \Big(\underbrace{1 \times a'([Z]_j)}_{\dfrac{\partial \varphi_j}{\partial [B]_j}} \Big) \times a'([Z']_i) \\ & = a'([Z']_i) \times \dfrac{\partial \varphi_j}{\partial [B]_j} \end{align*}$$

# Réseau à 3 couches

$$\begin{align*} X \in \mathbb R^m \quad & \underset{\displaystyle \varphi^{(1)}(\;\cdot\;, W^{(1)}, B^{(1)})}{\longmapsto} & X^{(1)} \in \mathbb R^n \\\\ & \underset{\displaystyle \varphi^{(2)}(\;\cdot\;, W^{(2)}, B^{(2)})}{\longmapsto} & X^{(2)} \in \mathbb R^p \\\\ & \underset{\displaystyle \varphi^{(3)}(\;\cdot\;, W^{(3)}, B^{(3)})}{\longmapsto} & Y \in \mathbb R^q \\\\ & \underset{\displaystyle \;\quad\quad L(\;\cdot\;, \tilde Y) \,\quad\quad}{\longmapsto} & \ell \in \mathbb R \end{align*}$$

$$
\begin{align*}

L(Y, \tilde Y) & = \dfrac{1}{2} \sum\limits_{k=1}^q ([Y]_k - [\tilde Y]_k)^2 \\

& = \dfrac{1}{2} \sum\limits_{i=1}^q \Bigg(\sum\limits_{j=1}^p \Big([W^{(3)}]_{i,j} [X^{(2)}]_j + [B^{(3)}]_i\Big) - [\tilde Y]_i\Bigg)^2 \\

& = \dfrac{1}{2} \sum\limits_{i=1}^q \Bigg(\sum\limits_{j=1}^p \Bigg([W^{(3)}]_{i,j} \sum\limits_{k=1}^n \Big([W^{(2)}]_{j,k} [X^{(1)}]_k + [B^{(2)}]_j\Big) + [B^{(3)}]_i\Bigg) - [\tilde Y]_i\Bigg)^2 \\

& = \dfrac{1}{2} \sum\limits_{i=1}^q \Bigg(\sum\limits_{j=1}^p \Bigg([W^{(3)}]_{i,j} \sum\limits_{k=1}^n \Bigg([W^{(2)}]_{j,k} \sum\limits_{l=1}^m \Big([W^{(1)}]_{k,l} [X]_l + [B^{(1)}]_k\Big) + [B^{(3)}]_i\Bigg) + [B^{(2)}]_j\Bigg) + [B^{(3)}]_i\Bigg) - [\tilde Y]_i\Bigg)^2

\end{align*}
$$
