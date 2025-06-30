import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Union


class Algorithm(ABC):
    r"""
    Solves the problem:

    .. math::
       \text{find } y \in H \text{ such that } 0 \in \sum_{i \in I_{\text{func}}} \partial f_i(y) + \sum_{i \in I_{\text{op}}} G_i(y)

    with the state-space representation:

    .. math::
       x^{k+1} = (A_k \otimes I)x^k + (B_k \otimes I)u^k \quad \text{and} \quad
       y^k = (C_k \otimes I)x^k + (D_k \otimes I)u^k

    where:

    - **n** is the dimension of :math:`x`.
    - **m** is the total number of components (indices :math:`1,\dots, m`), split into
      :math:`I_{\text{func}}` and :math:`I_{\text{op}}`.
    - For each :math:`i`, **\bar{m}_i** is the number of evaluations and

      .. math::
         \bar{m} = \sum_{i=1}^{m} \bar{m}_i.
    """

    def __init__(self, n: int, m: int, m_bar_is: List[int],
                 I_func: List[int], I_op: List[int]):
        # Basic validations
        if n < 1:
            raise ValueError("n must be at least 1")
        if m < 1:
            raise ValueError("m must be at least 1")
        if m != len(m_bar_is):
            raise ValueError("m must equal the length of m_bar_is")
        if not set(I_func).isdisjoint(I_op):
            raise ValueError("I_func and I_op must be disjoint")
        if set(I_func).union(I_op) != set(range(1, m + 1)):
            raise ValueError("I_func and I_op must cover {1,…, m}")
        if I_func and any(I_func[i] >= I_func[i + 1] for i in range(len(I_func) - 1)):
            raise ValueError("I_func must be in increasing order")
        if I_op and any(I_op[i] >= I_op[i + 1] for i in range(len(I_op) - 1)):
            raise ValueError("I_op must be in increasing order")

        self.n = n                      # Dimension of x.
        self.m = m                      # Total components.
        self.m_bar_is = m_bar_is        # Evaluations per component (\bar{m}_i).
        self.m_bar = sum(m_bar_is)      # Total evaluations (\bar{m}).
        self.I_func = I_func            # Functional indices (I_{\text{func}}).
        self.I_op = I_op                # Operator indices (I_{\text{op}}).
        self.m_func = len(I_func)       # Count of functional components.
        self.m_op = len(I_op)           # Count of operator components.
        self.m_bar_func = sum(m_bar_is[i - 1] for i in I_func) if I_func else 0  # \bar{m}_{\text{func}}
        self.m_bar_op = sum(m_bar_is[i - 1] for i in I_op) if I_op else 0          # \bar{m}_{\text{op}}
        # Mapping for functional indices (for F matrices)
        self.kappa = {I_func[i]: i + 1 for i in range(self.m_func)} if I_func else {}

    @abstractmethod
    def get_ABCD(self, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Return the system matrices (A, B, C, D) for iteration k.

        **Dimensions:**

          - **A**: (n x n)
          - **B**: (n x \bar{m})
          - **C**: (\bar{m} x n)
          - **D**: (\bar{m} x \bar{m})

        :param k: Iteration index.
        :type k: int
        :returns: A tuple (A, B, C, D) of numpy arrays.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        pass

    def get_AsBsCsDs(self, k_min: int, k_max: int
                      ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        r"""
        Return a dictionary mapping each iteration index k to the tuple (A, B, C, D)
        for k_min ≤ k ≤ k_max.

        :param k_min: The minimum iteration index.
        :type k_min: int
        :param k_max: The maximum iteration index.
        :type k_max: int
        :returns: A dictionary where keys are iteration indices and values are tuples (A, B, C, D).
        :rtype: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        """
        if k_min < 0 or k_max < k_min:
            raise ValueError("Require 0 <= k_min <= k_max")
        return {k: self.get_ABCD(k) for k in range(k_min, k_max + 1)}

    # --- U MATRICES ---
    def _generate_U(self, k_min: int, k_max: int, k: int = None, star: bool = False) -> np.ndarray:
        r"""
        Generate a U matrix.

        The total number of columns is given by:

        .. math::
           n + ((k_{max} - k_{min} + 1) \cdot \bar{m} + m).

        - If ``star=True``, returns U_{\text{star}} defined as:

          .. math::
             U_{\text{star}} = \begin{bmatrix}
             \mathbf{0}_{m \times \left(n + ((k_{max} - k_{min} + 1) \cdot \bar{m})\right)} &
             N &
             \mathbf{0}_{m \times 1}
             \end{bmatrix},

          where

          .. math::
             N = \begin{bmatrix} I_{m-1} \\ -\mathbf{1}_{1 \times (m-1)} \end{bmatrix}.

        - If ``star=False`` and k is provided (with k_min ≤ k ≤ k_max), returns:

          .. math::
             U_k = \begin{bmatrix}
             \mathbf{0}_{\bar{m} \times \left(n + (k - k_{min})\bar{m}\right)} &
             I_{\bar{m}} &
             \mathbf{0}_{\bar{m} \times \left((k_{max} - k)\bar{m} + m\right)}
             \end{bmatrix}.

        :param k_min: The minimum iteration index.
        :type k_min: int
        :param k_max: The maximum iteration index.
        :type k_max: int
        :param k: (Optional) The current iteration index (required if star is False).
        :type k: int, optional
        :param star: If True, generate the star matrix.
        :type star: bool
        :returns: The generated U matrix.
        :rtype: np.ndarray
        """
        total_cols = self.n + ((k_max - k_min + 1) * self.m_bar + self.m)
        if star:
            if self.m > 1:
                N = np.vstack([np.eye(self.m - 1), -np.ones((1, self.m - 1))])
                return np.hstack([
                    np.zeros((self.m, self.n + (k_max - k_min + 1) * self.m_bar)),
                    N,
                    np.zeros((self.m, 1))
                ])
            else:
                return np.zeros((self.m, total_cols))
        else:
            if k is None:
                raise ValueError("When star is False, k must be provided")
            if k < k_min or k > k_max:
                raise ValueError("k must be between k_min and k_max")
            left = np.zeros((self.m_bar, self.n + (k - k_min) * self.m_bar))
            ident = np.eye(self.m_bar)
            right = np.zeros((self.m_bar, (k_max - k) * self.m_bar + self.m))
            return np.hstack([left, ident, right])

    def get_Us(self, k_min: int, k_max: int) -> Dict[Any, np.ndarray]:
        r"""
        Return a dictionary of U matrices for iterations k_min ≤ k ≤ k_max, including the star matrix.

        :param k_min: The minimum iteration index.
        :type k_min: int
        :param k_max: The maximum iteration index.
        :type k_max: int
        :returns: A dictionary where keys are iteration indices (and 'star') and values are U matrices.
        :rtype: Dict[Any, np.ndarray]
        """
        if k_min < 0 or k_max < k_min:
            raise ValueError("Require 0 <= k_min <= k_max")
        Us = {}
        for k in range(k_min, k_max + 1):
            Us[k] = self._generate_U(k_min, k_max, k=k, star=False)
        Us['star'] = self._generate_U(k_min, k_max, star=True)
        return Us

    # --- Y MATRICES ---
    def _generate_Y(self,
                    sys_mats: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                    k_min: int, k_max: int, k: int = None, star: bool = False) -> np.ndarray:
        r"""
        Generate the output matrix Y using system matrices from sys_mats.

        The total number of columns is:

        .. math::
           n + (k_{max} - k_{min} + 1) \cdot \bar{m} + m.

        - If ``star=True``, returns Y_{\text{star}} defined as:

          .. math::
             Y_{\text{star}} = \begin{bmatrix} \mathbf{0}_{m \times (total\_cols - 1)} & \mathbf{1}_{m \times 1} \end{bmatrix}.

        - Otherwise, if star is False then k must be provided:
          
          - If :math:`k = k_{min}`, then

            .. math::
               Y_{k_{min}} = \begin{bmatrix} C_{k_{min}} & D_{k_{min}} & \mathbf{0} \end{bmatrix},

            where the zeros block has shape :math:`(\bar{m}, ((k_{max} - k_{min}) \cdot \bar{m} + m))`.

          - If :math:`k = k_{min} + 1`, then

            .. math::
               Y_{k_{min}+1} = \begin{bmatrix} C_{k_{min}+1} A_{k_{min}} & C_{k_{min}+1} B_{k_{min}} & D_{k_{min}+1} & \mathbf{0} \end{bmatrix},

            with the zeros block of shape :math:`(\bar{m}, ((k_{max} - k_{min} - 1) \cdot \bar{m} + m))`.

          - If :math:`k \ge k_{min} + 2`, then construct Y_k by:
            - Block 1: :math:`C_k (A_{k-1} \cdots A_{k_{min}})`,
            - For each :math:`j` from :math:`k_{min}` to :math:`k-2`, the next block is
              :math:`C_k (A_{k-1} \cdots A_{j+1}) B_j`,
            - Then block: :math:`C_k B_{k-1}`,
            - Followed by :math:`D_k` and a zeros block of shape :math:`(\bar{m}, ((k_{max} - k) \cdot \bar{m} + m))`.

        :param sys_mats: Dictionary mapping each iteration index to (A, B, C, D).
        :type sys_mats: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        :param k_min: The minimum iteration index.
        :type k_min: int
        :param k_max: The maximum iteration index.
        :type k_max: int
        :param k: (Optional) The current iteration index (required if star is False).
        :type k: int, optional
        :param star: If True, generate Y_{\text{star}}.
        :type star: bool
        :returns: The generated Y matrix.
        :rtype: np.ndarray
        """
        total_cols = self.n + (k_max - k_min + 1) * self.m_bar + self.m

        if star:
            return np.hstack([
                np.zeros((self.m, total_cols - 1)),
                np.ones((self.m, 1))
            ])
        if k is None:
            raise ValueError("When star is False, k must be provided")

        if k == k_min:
            C_kmin = sys_mats[k_min][2]
            D_kmin = sys_mats[k_min][3]
            zeros_blk = np.zeros((self.m_bar, (k_max - k_min) * self.m_bar + self.m))
            return np.hstack([C_kmin, D_kmin, zeros_blk])
        elif k == k_min + 1:
            C_next = sys_mats[k][2]
            A_kmin = sys_mats[k_min][0]
            B_kmin = sys_mats[k_min][1]
            D_next = sys_mats[k][3]
            zeros_blk = np.zeros((self.m_bar, (k_max - k_min - 1) * self.m_bar + self.m))
            block1 = C_next @ A_kmin
            block2 = C_next @ B_kmin
            return np.hstack([block1, block2, D_next, zeros_blk])
        else:
            blocks = []
            C_k = sys_mats[k][2]
            prod = C_k.copy()
            for i in reversed(range(k_min, k)):
                prod = prod @ sys_mats[i][0]
            blocks.append(prod)
            for j in range(k_min, k - 1):
                prod_B = np.eye(self.n)
                for i in reversed(range(j + 1, k)):
                    prod_B = prod_B @ sys_mats[i][0]
                blocks.append(C_k @ prod_B @ sys_mats[j][1])
            blocks.append(C_k @ sys_mats[k - 1][1])
            D_k = sys_mats[k][3]
            blocks.append(D_k)
            zeros_blk = np.zeros((self.m_bar, (k_max - k) * self.m_bar + self.m))
            blocks.append(zeros_blk)
            return np.hstack(blocks)

    def get_Ys(self, k_min: int, k_max: int) -> Dict[Any, np.ndarray]:
        r"""
        Return a dictionary of Y matrices for iterations k_min ≤ k ≤ k_max, including Y_{\text{star}}.

        :param k_min: The minimum iteration index.
        :type k_min: int
        :param k_max: The maximum iteration index.
        :type k_max: int
        :returns: A dictionary where keys are iteration indices (and 'star') and values are Y matrices.
        :rtype: Dict[Any, np.ndarray]
        """
        if k_min < 0 or k_max < k_min:
            raise ValueError("Require 0 <= k_min <= k_max")
        sys_mats = self.get_AsBsCsDs(k_min, k_max)
        Ys = {}
        for k in range(k_min, k_max + 1):
            Ys[k] = self._generate_Y(sys_mats, k_min, k_max, k=k, star=False)
        Ys['star'] = self._generate_Y(sys_mats, k_min, k_max, star=True)
        return Ys

    # --- X MATRICES ---
    def _generate_X_k(self, sys_mats: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                        k: int, k_min: int, k_max: int) -> np.ndarray:
        r"""
        Generate the state matrix X_k for iterations k_min ≤ k ≤ k_max+1.

        - For :math:`k = k_{min}`, X_{k_{min}} is given by
          .. math:: [I_n, \mathbf{0}],
          where the zeros block has the appropriate number of columns.
        - For :math:`k = k_{min} + 1`, X_{k_{min}+1} is given by
          .. math:: [A_{k_{min}}, B_{k_{min}}, \mathbf{0}],
          with a zeros block of appropriate size.
        - For :math:`k \ge k_{min} + 2`, X_k is constructed as:

          .. math::
             X_k = \left[ A_{k-1}\cdots A_{k_{min}},\; (A_{k-1}\cdots A_{j+1}) B_j \text{ for } j=k_{min},\dots,k-2,\; B_{k-1},\; \mathbf{0} \right].

        :param sys_mats: Dictionary mapping each iteration index to (A, B, C, D).
        :type sys_mats: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        :param k: The current iteration index.
        :type k: int
        :param k_min: The minimum iteration index.
        :type k_min: int
        :param k_max: The maximum iteration index.
        :type k_max: int
        :returns: The generated X_k matrix.
        :rtype: np.ndarray
        """
        total_cols = self.n + (k_max - k_min + 1) * self.m_bar + self.m

        if k < k_min or k > k_max + 1:
            raise ValueError("k must be in [k_min, k_max+1]")

        if k == k_min:
            return np.hstack([np.eye(self.n), np.zeros((self.n, total_cols - self.n))])
        if k == k_min + 1:
            A, B, _, _ = sys_mats[k_min]
            return np.hstack([A, B, np.zeros((self.n, total_cols - self.n - self.m_bar))])
        
        parts = []
        prod = np.eye(self.n)
        for i in reversed(range(k_min, k)):
            prod = prod @ sys_mats[i][0]
        parts.append(prod)
        for j in range(k_min, k - 1):
            prod_B = np.eye(self.n)
            for i in reversed(range(j + 1, k)):
                prod_B = prod_B @ sys_mats[i][0]
            parts.append(prod_B @ sys_mats[j][1])
        parts.append(sys_mats[k - 1][1])
        zeros_blk = np.zeros((self.n, (k_max + 1 - k) * self.m_bar + self.m))
        parts.append(zeros_blk)
        return np.hstack(parts)

    def get_Xs(self, k_min: int, k_max: int) -> Dict[int, np.ndarray]:
        r"""
        Return a dictionary mapping each iteration index k (for k_min ≤ k ≤ k_max+1) to the corresponding X_k matrix.

        The X_k matrices are constructed using system matrices retrieved via
        :meth:`get_AsBsCsDs` and the helper method :meth:`_generate_X_k`.

        :param k_min: The minimum iteration index.
        :type k_min: int
        :param k_max: The maximum iteration index.
        :type k_max: int
        :returns: A dictionary with keys as iteration indices and values as X_k matrices.
        :rtype: Dict[int, np.ndarray]
        """
        if k_min < 0 or k_max < k_min:
            raise ValueError("Require 0 <= k_min <= k_max")
        sys_mats = self.get_AsBsCsDs(k_min, k_max)
        Xs = {}
        for k in range(k_min, k_max + 2):
            Xs[k] = self._generate_X_k(sys_mats, k, k_min, k_max)
        return Xs

    # --- PROJECTION MATRICES (P) ---
    def get_Ps(self) -> Dict[Tuple[Any, Any], np.ndarray]:
        r"""
        Return a dictionary of projection matrices P.

        For each component index :math:`i \in \{1,\dots, m\}` and for each evaluation index
        :math:`j \in \{1,\dots, \bar{m}_i\}`, the projection matrix :math:`P_{(i,j)}` is a
        1 x \bar{m} row vector with a 1 at the (offset+j)-th position, where

        .. math::
           \text{offset} = \sum_{l=1}^{i-1} \bar{m}_l.

        Additionally, :math:`P_{(i,\text{'star'})}` is a 1 x m row vector with a 1 in the i-th position.

        :returns: A dictionary where keys are tuples (i, j) or (i, 'star') and values are projection matrices.
        :rtype: Dict[Tuple[Any, Any], np.ndarray]
        """
        Ps = {}
        for i in range(1, self.m + 1):
            offset = sum(self.m_bar_is[:i - 1])
            for j in range(1, self.m_bar_is[i - 1] + 1):
                vec = np.zeros((1, self.m_bar))
                vec[0, offset + j - 1] = 1
                Ps[(i, j)] = vec
            star_vec = np.zeros((1, self.m))
            star_vec[0, i - 1] = 1
            Ps[(i, 'star')] = star_vec
        return Ps

    # --- F MATRICES (for functional components) ---
    def _generate_F(self, i: int, j: int = None, k: int = None,
                    star: bool = False, k_min: int = 0, k_max: int = 0) -> np.ndarray:
        r"""
        Generate a single row of the F matrix for a functional component indexed by i.

        The overall F row has dimension

        .. math::
           \left( 1, \, ((k_{max} - k_{min} + 1) \cdot \bar{m}_{\text{func}} + m_{\text{func}}) \right).

        - For the non-star case (i.e. :math:`F_{(i,j,k)}`), a 1 is placed at the location corresponding to
          the j-th evaluation of component i, shifted by the contributions of all preceding functional components
          and by (k - k_{min}) blocks of size \(\bar{m}_{\text{func}}\).
        - For the star case (i.e. :math:`F_{(i,\text{'star'},\text{'star'})}`), a 1 is placed in the last
          m_{\text{func}} entries, specifically at index

          .. math::
             (k_{max} - k_{min} + 1) \cdot \bar{m}_{\text{func}} + (\kappa[i]-1).

        :param i: Functional component index (must be in :attr:`I_func`).
        :type i: int
        :param j: Evaluation index for the non-star case.
        :type j: int, optional
        :param k: Iteration index for the non-star case (must be in [k_min, k_max]).
        :type k: int, optional
        :param star: If True, generate the star row.
        :type star: bool
        :param k_min: The minimum iteration index.
        :type k_min: int
        :param k_max: The maximum iteration index.
        :type k_max: int
        :returns: A 1 x total_dim numpy array representing the generated F row.
        :rtype: np.ndarray
        """
        total_dim = ((k_max - k_min + 1) * self.m_bar_func + self.m_func)

        if not self.I_func or i not in self.I_func:
            raise ValueError(f"i must be in I_func. Got i = {i}")
        idx = self.kappa[i]  # 1-indexed

        if star:
            if j is not None or k is not None:
                raise ValueError("For star F matrices, do not supply j or k")
            F_star = np.zeros((1, total_dim))
            F_star[0, (k_max - k_min + 1) * self.m_bar_func + idx - 1] = 1
            return F_star
        else:
            if j is None or k is None:
                raise ValueError("For non-star F matrices, both j and k must be provided")
            if not (k_min <= k <= k_max):
                raise ValueError(f"k must be in [{k_min}, {k_max}]. Got k = {k}")
            max_j = self.m_bar_is[i - 1]
            if not (1 <= j <= max_j):
                raise ValueError(f"For i = {i}, j must be in [1, {max_j}]. Got j = {j}")
            F_nonstar = np.zeros((1, total_dim))
            offset = sum(self.m_bar_is[l - 1] for l in self.I_func if l < i)
            start_idx = self.m_bar_func * (k - k_min) + offset
            F_nonstar[0, start_idx + j - 1] = 1
            return F_nonstar

    def get_Fs(self, k_min: int, k_max: int) -> Dict[Tuple[Any, Any, Any], np.ndarray]:
        r"""
        Return a dictionary of F matrices for all functional components.

        The dictionary keys are defined as follows:

        - For non-star F matrices: keys are of the form (i, j, k), where

          - i is in :attr:`I_func`,
          - j is in [1, m_bar_is[i-1]],
          - k is in [k_min, k_max].
          
        - For star F matrices: keys are of the form (i, 'star', 'star') for each i in :attr:`I_func`.

        :param k_min: The minimum iteration index.
        :type k_min: int
        :param k_max: The maximum iteration index.
        :type k_max: int
        :returns: A dictionary mapping keys to the corresponding F row matrices.
        :rtype: Dict[Tuple[Any, Any, Any], np.ndarray]
        """
        if k_min < 0 or k_max < k_min:
            raise ValueError("Require 0 <= k_min <= k_max")
        Fs = {}
        for k in range(k_min, k_max + 1):
            for i in self.I_func:
                for j in range(1, self.m_bar_is[i - 1] + 1):
                    Fs[(i, j, k)] = self._generate_F(i, j, k, star=False,
                                                      k_min=k_min, k_max=k_max)
        for i in self.I_func:
            Fs[(i, 'star', 'star')] = self._generate_F(i, star=True,
                                                         k_min=k_min, k_max=k_max)
        return Fs

    # --- LIFTED CONSTRAINT MATRICES ---
    def compute_E(self, i: int, pairs: List[Tuple[Any, Any]],
                  k_min: int, k_max: int) -> np.ndarray:
        r"""
        Compute the E matrix for component i using a list of (j, k) pairs.

        The E matrix is defined as:

        .. math::
           E^{k_{min}, k_{max}}_{(i, j_1, k_1, \dots, j_{n_{i,o}}, k_{n_{i,o}})} =
           \begin{bmatrix}
           P_{(i,j_1)}Y_{k_1} \\
           \vdots \\
           P_{(i,j_{n_{i,o}})}Y_{k_{n_{i,o}}} \\
           P_{(i,j_1)}U_{k_1} \\
           \vdots \\
           P_{(i,j_{n_{i,o}})}U_{k_{n_{i,o}}}
           \end{bmatrix}.

        The resulting matrix has dimensions

        .. math::
           2 \cdot (\text{number of pairs}) \times \left[ n + (k_{max} - k_{min} + 1) \cdot \bar{m} + m \right].

        :param i: The component index.
        :type i: int
        :param pairs: A list of (j, k) pairs. For non-star pairs, j must be an integer in
                      [1, m_bar_is[i-1]] and k must lie in [k_min, k_max]. For the star case,
                      the pair should be ('star', 'star').
        :type pairs: List[Tuple[Any, Any]]
        :param k_min: The minimum iteration index.
        :type k_min: int
        :param k_max: The maximum iteration index.
        :type k_max: int
        :returns: The computed E matrix.
        :rtype: np.ndarray
        """
        for pair in pairs:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError("Each element in pairs must be a tuple of two elements (j, k).")
        if k_min < 0 or k_max < k_min:
            raise ValueError("Invalid iteration bounds: ensure 0 <= k_min <= k_max.")

        Ps = self.get_Ps()
        Ys = self.get_Ys(k_min, k_max)
        Us = self.get_Us(k_min, k_max)

        Y_blocks = []
        U_blocks = []

        for (j, k) in pairs:
            if j == 'star' and k == 'star':
                P = Ps[(i, 'star')]
                Y_blk = P @ Ys['star']
                U_blk = P @ Us['star']
            else:
                if not (isinstance(j, int) and isinstance(k, int)):
                    raise ValueError("For non-star pairs, (j, k) must be integers.")
                if not (1 <= j <= self.m_bar_is[i - 1]):
                    raise ValueError(f"For i = {i}, j must be in [1, {self.m_bar_is[i - 1]}].")
                if not (k_min <= k <= k_max):
                    raise ValueError(f"k must be in [{k_min}, {k_max}].")
                P = Ps[(i, j)]
                Y_blk = P @ Ys[k]
                U_blk = P @ Us[k]
            Y_blocks.append(Y_blk)
            U_blocks.append(U_blk)

        return np.vstack(Y_blocks + U_blocks)

    def compute_W(self, i: int, pairs: List[Tuple[Any, Any]],
                  k_min: int, k_max: int, M: np.ndarray) -> np.ndarray:
        r"""
        Compute the W matrix for component i.

        The W matrix is given by:

        .. math::
           W = E^T M E,

        where E is computed via :meth:`compute_E`.

        :param i: The component index.
        :type i: int
        :param pairs: A list of (j, k) pairs. For non-star pairs, each pair must be (int, int)
                      with j in the valid range for component i and k in [k_min, k_max]. For star rows,
                      the pair should be ('star', 'star').
        :type pairs: List[Tuple[Any, Any]]
        :param k_min: The minimum iteration index.
        :type k_min: int
        :param k_max: The maximum iteration index.
        :type k_max: int
        :param M: A symmetric matrix of dimension [2*(number of pairs) x 2*(number of pairs)].
        :type M: np.ndarray
        :returns: The computed W matrix.
        :rtype: np.ndarray
        :raises ValueError: If any input conditions are not met.
        """
        for pair in pairs:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError("Each element in pairs must be a tuple of two elements (j, k).")
        if k_min < 0 or k_max < k_min:
            raise ValueError("Invalid iteration bounds: ensure 0 <= k_min <= k_max.")
        exp_dim = 2 * len(pairs)
        if M.ndim != 2 or M.shape != (exp_dim, exp_dim):
            raise ValueError(f"M must be a square matrix of dimension [{exp_dim} x {exp_dim}].")
        if not np.allclose(M, M.T, atol=1e-8):
            raise ValueError("M must be symmetric.")

        E = self.compute_E(i, pairs, k_min, k_max)
        return E.T @ M @ E

    def compute_F_aggregated(self, i: int, pairs: List[Union[Tuple[int, int], Tuple[str, str]]],
                             k_min: int, k_max: int, a: np.ndarray) -> np.ndarray:
        r"""
        Compute the aggregated F vector for component i.

        The aggregated F vector is computed as:

        .. math::
           F_{\text{aggregated}} = \begin{bmatrix} (F_{(i,j_1,k_1)})^T & \cdots & (F_{(i,j_n,k_n)})^T \end{bmatrix} a,

        where each F row is obtained from the F matrices (via :meth:`get_Fs`), transposed, and
        horizontally stacked. The resulting matrix has shape (total_dim, number of pairs) and is then
        multiplied by the weight vector :math:`a` (a 1D array of length equal to the number of pairs) to
        yield a column vector of shape (total_dim, 1).

        Here,

        .. math::
           \text{total\_dim} = n + (k_{max} - k_{min} + 1) \cdot \bar{m}_{\text{func}} + m_{\text{func}}.

        :param i: The component index (should be in :attr:`I_func`).
        :type i: int
        :param pairs: A list of (j, k) pairs. For non-star cases, each pair must be (int, int) with
                      j in [1, m_bar_is[i-1]] and k in [k_min, k_max]. For the star case, the pair
                      should be ('star', 'star').
        :type pairs: List[Union[Tuple[int, int], Tuple[str, str]]]
        :param k_min: The minimum iteration index.
        :type k_min: int
        :param k_max: The maximum iteration index.
        :type k_max: int
        :param a: A 1D numpy array of weights with length equal to the number of pairs.
        :type a: np.ndarray
        :returns: The aggregated F vector as a column vector.
        :rtype: np.ndarray
        :raises ValueError: If any input conditions are not met.
        """
        if i not in self.I_func:
            raise ValueError(f"Component index i must be in I_func. Got i = {i}.")

        total_dim = (k_max - k_min + 1) * self.m_bar_func + self.m_func

        for pair in pairs:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError("Each element in pairs must be a tuple of two elements (j, k).")
            j, k = pair
            if (j == 'star' or k == 'star') and not (j == 'star' and k == 'star'):
                raise ValueError("For star F matrices, both j and k must be 'star'.")
            if j != 'star' and k != 'star':
                if not (isinstance(j, int) and isinstance(k, int)):
                    raise ValueError("For non-star F matrices, both j and k must be integers.")
                if not (k_min <= k <= k_max):
                    raise ValueError(f"k must be in [{k_min}, {k_max}]. Got k = {k}.")
                if not (1 <= j <= self.m_bar_is[i - 1]):
                    raise ValueError(f"For component i = {i}, j must be in [1, {self.m_bar_is[i - 1]}]. Got j = {j}.")
        if not pairs:
            raise ValueError("Pairs list must be nonempty.")

        if not (isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == len(pairs)):
            raise ValueError("a must be a 1D numpy array with length equal to the number of pairs.")

        Fs_dict = self.get_Fs(k_min, k_max)
        F_cols = []
        for (j, k) in pairs:
            key = (i, 'star', 'star') if (j == 'star' and k == 'star') else (i, j, k)
            if key not in Fs_dict:
                raise ValueError(f"Key {key} not found in F matrices.")
            F_cols.append(Fs_dict[key].T)
        F_stack = np.hstack(F_cols)
        aggregated_F = F_stack @ a
        return aggregated_F.reshape(total_dim, 1)
