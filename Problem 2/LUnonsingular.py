"""
Permuted LU Decomposition — Nonsingular Case  (Theorem 1.11, p.28)
====================================================================
A NONSINGULAR matrix A may need row swaps during elimination.
These row swaps are captured in a permutation matrix P, giving:

    P · A = L · U          (equation 1.33)

where:
  P = permutation matrix   (identity matrix with some rows swapped)
  L = lower unitriangular  (1's on diagonal; multipliers below diagonal)
  U = upper triangular     (nonzero pivots on diagonal)

WHY do we need P?
-----------------
Nonsingular matrices can have zeros on the diagonal mid-elimination.
We can't use a zero as a pivot (we'd divide by zero).
The fix: swap in a row from below that has a nonzero entry in that column.
P records exactly which rows were swapped and in what order.

SOLVING with PA = LU:
---------------------
Start from  A x = b.
Multiply both sides on the left by P:

    P A x = P b    →    L U x = P b

Let  c = U x.  Then:

  Step 1 — Forward Substitution:   L c = P b  (solve for c, top-to-bottom)
  Step 2 — Back Substitution:      U x = c    (solve for x, bottom-to-top)

KEY RULE when updating L during a row swap:
-------------------------------------------
When we swap rows k and j in A, we also:
  - Swap the SAME two rows in P
  - Swap only the ALREADY-COMPUTED entries below the diagonal in L
    (the entries in columns 0..j-1 for rows j and k)
  - Entries on and above the diagonal of L are NOT swapped

This asymmetry is what makes the factorization valid.

Example (textbook p.27–28):
    A = [[0, 2, 1],       P swaps rows 0 and 1
         [2, 6, 1],
         [1, 1, 4]]

    PA = LU  where  P = [[0,1,0],[1,0,0],[0,0,1]]
"""

import numpy as np


# ── STEP 0 — PERMUTED LU FACTORIZATION ───────────────────────────────────────

def permuted_lu_factorize(A):
    """
    Decompose a NONSINGULAR matrix A into P, L, U such that  P · A = L · U.

    Algorithm (from textbook p.28–30):
      Initialize  P = I,  L = I,  work on a copy of A (call it M).

      For each pivot column j:
        (a) If M[j,j] = 0, find a row k > j where M[k,j] ≠ 0.
            - Swap rows j and k in M  (the working matrix)
            - Swap rows j and k in P  (to record the row interchange)
            - Swap ONLY the already-filled entries (columns 0..j-1)
              in rows j and k of L  (do NOT touch diagonal or above)
        (b) Perform elimination: for each row i > j, compute multiplier,
            store in L[i,j], and zero out M[i,j].

    Parameters
    ----------
    A : n×n nonsingular matrix

    Returns
    -------
    P : n×n permutation matrix
    L : n×n lower unitriangular matrix
    U : n×n upper triangular matrix
    """
    n = A.shape[0]
    M = A.copy()           # working copy — will become U
    P = np.eye(n)          # permutation tracker — starts as identity
    L = np.eye(n)          # multiplier recorder — starts as identity (1's on diagonal)

    for j in range(n):     # j = current pivot column

        # ── (a) Pivoting: swap rows if the diagonal entry is zero ────────────

        if M[j, j] == 0:

            # Find a nonzero entry in column j below row j
            pivot_row = None
            for k in range(j + 1, n):
                if M[k, j] != 0:
                    pivot_row = k
                    break

            if pivot_row is None:
                raise ValueError(
                    f"Singular matrix: column {j} is entirely zero from row {j} downward. "
                    "No unique solution exists."
                )

            print(f"  [Pivot] Swapping row {j} ↔ row {pivot_row}")

            # Swap rows in the working matrix M
            M[[j, pivot_row]] = M[[pivot_row, j]]

            # Swap the SAME rows in P (records the permutation)
            P[[j, pivot_row]] = P[[pivot_row, j]]

            # Swap only the ALREADY-COMPUTED entries of L
            # (columns 0 through j-1, which lie below the diagonal for these rows)
            # Entries at column j and beyond haven't been filled yet, so leave them alone.
            L[[j, pivot_row], :j] = L[[pivot_row, j], :j]

        # ── (b) Elimination: zero out entries below the pivot ────────────────

        for i in range(j + 1, n):
            if M[j, j] == 0:
                break                       # safety check (shouldn't happen now)
            lij = M[i, j] / M[j, j]
            L[i, j] = lij                  # store multiplier in L
            M[i, :] -= lij * M[j, :]       # zero out M[i,j]

    U = M   # M has been reduced to upper triangular form
    return P, L, U


# ── STEP 1 — FORWARD SUBSTITUTION: solve L c = Pb ─────────────────────────────

def forward_substitution(L, b):
    """
    Solve  L c = b  where L is lower unitriangular.
    (Here b should already be the permuted right-hand side P·b.)

    Works top-to-bottom. L[i,i] = 1 always, so no division needed.
    """
    n = len(b)
    c = np.zeros(n)
    for i in range(n):
        c[i] = b[i]
        for j in range(i):
            c[i] -= L[i, j] * c[j]
    return c


# ── STEP 2 — BACK SUBSTITUTION: solve U x = c ─────────────────────────────────

def back_substitution(U, c):
    """
    Solve  U x = c  where U is upper triangular.
    Works bottom-to-top.
    """
    n = len(c)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = c[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    return x


# ── FULL PIPELINE: A x = b  via  PA = LU ─────────────────────────────────────

def solve_permuted_lu(A, b, verbose=True):
    """
    Solve  A x = b  for a NONSINGULAR matrix A using permuted LU decomposition.

    Full pipeline:
      1. Factorize:    A  →  P, L, U    (PA = LU)
      2. Permute rhs:  Pb  (apply same row swaps to b)
      3. Forward sub:  L c = Pb
      4. Back sub:     U x = c

    Parameters
    ----------
    A : n×n nonsingular coefficient matrix
    b : right-hand side vector of length n

    Returns
    -------
    x : solution vector
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # ── Factorization ──
    P, L, U = permuted_lu_factorize(A)

    if verbose:
        print("P (permutation matrix — records row swaps):")
        print_matrix(P)
        print("\nL (lower unitriangular — stores multipliers):")
        print_matrix(L)
        print("\nU (upper triangular — pivots on diagonal):")
        print_matrix(U)
        print("\nVerification  P · A = L · U:")
        lhs = P @ A
        rhs = L @ U
        print("  P·A:")
        print_matrix(lhs)
        print("  L·U:")
        print_matrix(rhs)
        match = np.allclose(lhs, rhs)
        print(f"  PA = LU ? {'YES ✓' if match else 'NO ✗'}")

    # ── Apply P to b: permute the right-hand side ──
    Pb = P @ b
    if verbose:
        print(f"\nPermuted right-hand side  Pb = {np.round(Pb, 6)}")

    # ── Forward substitution: L c = Pb ──
    c = forward_substitution(L, Pb)
    if verbose:
        print(f"Forward substitution  L c = Pb  →  c = {np.round(c, 6)}")

    # ── Back substitution: U x = c ──
    x = back_substitution(U, c)
    return x


# ── HELPER ────────────────────────────────────────────────────────────────────

def print_matrix(M):
    for row in M:
        print("  [" + "  ".join(f"{v:8.4f}" for v in row) + " ]")


# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 66)
    print("  PERMUTED LU DECOMPOSITION — NONSINGULAR CASE  (PA = LU)")
    print("=" * 66)

    # ── Example 1: Textbook p.27 — needs one row swap ─────────────────────────
    print()
    print("Example 1 (textbook p.27 — zero in (1,1) position, one swap needed):")
    print("   0x + 2y +  z = 2")
    print("   2x + 6y +  z = 7")
    print("    x +  y + 4z = 3")
    print()

    A1 = np.array([
        [0, 2, 1],
        [2, 6, 1],
        [1, 1, 4]
    ], dtype=float)
    b1 = np.array([2, 7, 3], dtype=float)

    x1 = solve_permuted_lu(A1, b1)
    print(f"\nSolution: x={x1[0]:.6f}, y={x1[1]:.6f}, z={x1[2]:.6f}")
    print("\nVerification  A · x = b:")
    for i, (val, exp) in enumerate(zip(A1 @ x1, b1)):
        print(f"  Row {i+1}: {val:.6f}  (expected {exp})")

    # ── Example 2: Textbook p.28 Example 1.12 — 4×4 matrix ───────────────────
    print()
    print("─" * 66)
    print()
    print("Example 2 (textbook p.28–30, Example 1.12 — 4×4, multiple swaps):")
    print()

    A2 = np.array([
        [ 1,  2, -1,  0],
        [ 2,  4, -2, -1],
        [-3, -5,  6,  1],
        [-1,  2,  8, -2]
    ], dtype=float)
    b2 = np.array([1, 3, 0, -1], dtype=float)

    x2 = solve_permuted_lu(A2, b2)
    print(f"\nSolution: x={x2[0]:.6f}, y={x2[1]:.6f}, z={x2[2]:.6f}, w={x2[3]:.6f}")
    print("\nVerification  A · x = b:")
    for i, (val, exp) in enumerate(zip(A2 @ x2, b2)):
        print(f"  Row {i+1}: {val:.6f}  (expected {exp})")

    # ── Example 3: Regular matrix fed to permuted solver (no swap needed) ─────
    print()
    print("─" * 66)
    print()
    print("Example 3 — REGULAR matrix (no swap needed; PA=LU with P=I):")
    print("   2x +  y +  z = 1")
    print("   4x + 5y + 2z = 2")
    print("   2x − 2y + 0z = 2")
    print()

    A3 = np.array([
        [2,  1,  1],
        [4,  5,  2],
        [2, -2,  0]
    ], dtype=float)
    b3 = np.array([1, 2, 2], dtype=float)

    x3 = solve_permuted_lu(A3, b3)
    print(f"\nSolution: x={x3[0]:.6f}, y={x3[1]:.6f}, z={x3[2]:.6f}")
    print("\nVerification  A · x = b:")
    for i, (val, exp) in enumerate(zip(A3 @ x3, b3)):
        print(f"  Row {i+1}: {val:.6f}  (expected {exp})")

    # ── Example 4: Singular matrix — should raise an error ───────────────────
    print()
    print("─" * 66)
    print()
    print("Example 4 — SINGULAR matrix (to show error detection):")
    print("    x +  y = 1")
    print("   2x + 2y = 5   ← impossible (2× first equation ≠ 5)")
    print()

    A4 = np.array([[1, 1], [2, 2]], dtype=float)
    b4 = np.array([1, 5], dtype=float)
    try:
        solve_permuted_lu(A4, b4)
    except ValueError as e:
        print(f"  ERROR: {e}")
