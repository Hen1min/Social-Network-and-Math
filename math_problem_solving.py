import numpy as np

def build_laplace_system(k, left, right, bottom, top):
    n_side = k - 1
    n = n_side * n_side
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    def idx(i, j):
        return (i - 1) * n_side + (j - 1)

    h = 1.0 / k
    for i in range(1, k):
        y = i * h
        for j in range(1, k):
            x = j * h
            p = idx(i, j)
            A[p, p] = 4.0

            if j > 1:
                A[p, idx(i, j - 1)] = -1.0
            else:
                b[p] += left(y)

            if j < k - 1:
                A[p, idx(i, j + 1)] = -1.0
            else:
                b[p] += right(y)

            if i > 1:
                A[p, idx(i - 1, j)] = -1.0
            else:
                b[p] += bottom(x)

            if i < k - 1:
                A[p, idx(i + 1, j)] = -1.0
            else:
                b[p] += top(x)

    return A, b


def residual_ratio(A, b, x, r0_norm):
    r = b - A.dot(x)
    r_norm = np.linalg.norm(r)
    if r0_norm == 0.0:
        return r_norm, r_norm
    return r_norm / r0_norm, r_norm


def jacobi(A, b, x0, tol_ratio=1e-4, max_iterations=20000):
    n = len(b)
    x = x0.copy()
    d = np.diag(A)
    r0_norm = np.linalg.norm(b - A.dot(x))

    for it in range(1, max_iterations + 1):
        x_new = x.copy()
        for i in range(n):
            s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1 :], x[i + 1 :])
            x_new[i] = (b[i] - s) / d[i]
        x = x_new
        ratio, r_norm = residual_ratio(A, b, x, r0_norm)
        if ratio < tol_ratio:
            return {"x": x, "iterations": it, "ratio": ratio, "r_norm": r_norm, "converged": True}

    ratio, r_norm = residual_ratio(A, b, x, r0_norm)
    return {"x": x, "iterations": max_iterations, "ratio": ratio, "r_norm": r_norm, "converged": False}


def gauss_seidel(A, b, x0, tol_ratio=1e-4, max_iterations=20000):
    n = len(b)
    x = x0.copy()
    r0_norm = np.linalg.norm(b - A.dot(x))

    for it in range(1, max_iterations + 1):
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1 :], x[i + 1 :])
            x[i] = (b[i] - s1 - s2) / A[i, i]
        ratio, r_norm = residual_ratio(A, b, x, r0_norm)
        if ratio < tol_ratio:
            return {"x": x, "iterations": it, "ratio": ratio, "r_norm": r_norm, "converged": True}

    ratio, r_norm = residual_ratio(A, b, x, r0_norm)
    return {"x": x, "iterations": max_iterations, "ratio": ratio, "r_norm": r_norm, "converged": False}


def sor(A, b, x0, omega=1.5, tol_ratio=1e-4, max_iterations=20000):
    n = len(b)
    x = x0.copy()
    r0_norm = np.linalg.norm(b - A.dot(x))

    for it in range(1, max_iterations + 1):
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1 :], x[i + 1 :])
            x_gs = (b[i] - s1 - s2) / A[i, i]
            x[i] = (1.0 - omega) * x[i] + omega * x_gs
        ratio, r_norm = residual_ratio(A, b, x, r0_norm)
        if ratio < tol_ratio:
            return {"x": x, "iterations": it, "ratio": ratio, "r_norm": r_norm, "converged": True}

    ratio, r_norm = residual_ratio(A, b, x, r0_norm)
    return {"x": x, "iterations": max_iterations, "ratio": ratio, "r_norm": r_norm, "converged": False}


def steepest_descent(A, b, x0, tol_ratio=1e-4, max_iterations=20000):
    x = x0.copy()
    r = b - A.dot(x)
    r0_norm = np.linalg.norm(r)

    for it in range(1, max_iterations + 1):
        Ar = A.dot(r)
        denom = np.dot(r, Ar)
        if denom == 0.0:
            break
        alpha = np.dot(r, r) / denom
        x = x + alpha * r
        r = b - A.dot(x)
        r_norm = np.linalg.norm(r)
        ratio = r_norm if r0_norm == 0.0 else r_norm / r0_norm
        if ratio < tol_ratio:
            return {"x": x, "iterations": it, "ratio": ratio, "r_norm": r_norm, "converged": True}

    r_norm = np.linalg.norm(r)
    ratio = r_norm if r0_norm == 0.0 else r_norm / r0_norm
    return {"x": x, "iterations": max_iterations, "ratio": ratio, "r_norm": r_norm, "converged": False}


def conjugate_gradient(A, b, x0, tol_ratio=1e-4, max_iterations=20000):
    x = x0.copy()
    r = b - A.dot(x)
    p = r.copy()
    r0_norm = np.linalg.norm(r)
    rr_old = np.dot(r, r)

    for it in range(1, max_iterations + 1):
        Ap = A.dot(p)
        denom = np.dot(p, Ap)
        if denom == 0.0:
            break
        alpha = rr_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = np.dot(r, r)
        r_norm = np.sqrt(rr_new)
        ratio = r_norm if r0_norm == 0.0 else r_norm / r0_norm
        if ratio < tol_ratio:
            return {"x": x, "iterations": it, "ratio": ratio, "r_norm": r_norm, "converged": True}
        beta = rr_new / rr_old
        p = r + beta * p
        rr_old = rr_new

    r_norm = np.linalg.norm(r)
    ratio = r_norm if r0_norm == 0.0 else r_norm / r0_norm
    return {"x": x, "iterations": max_iterations, "ratio": ratio, "r_norm": r_norm, "converged": False}


def print_results(results):
    print("Method               Converged  Iterations  ResidualRatio")
    print("-" * 72)
    for name, res in results.items():
        print(
            f"{name:<20} {str(res['converged']):<10} {res['iterations']:<11} "
            f"{res['ratio']:<16.6e}"
        )


def print_solutions(results):
    print("\nFinal numerical solution of each method:")
    for name, res in results.items():
        x = res["x"]
        print(f"\n{name}")
        print("x =")
        print(np.array2string(x, precision=6, suppress_small=True))


def main():
    k = 6
    tol_ratio = 1e-4
    max_iterations = 20000

    # Dirichlet boundaries for demo: u=0 on left/right/bottom, u=1 on top.
    left = lambda y: 0.0
    right = lambda y: 0.0
    bottom = lambda x: 0.0
    top = lambda x: 1.0

    A, b = build_laplace_system(k, left, right, bottom, top)
    x0 = np.zeros_like(b)

    results = {
        "Jacobi": jacobi(A, b, x0, tol_ratio, max_iterations),
        "Gauss-Seidel": gauss_seidel(A, b, x0, tol_ratio, max_iterations),
        "SOR(omega=1.5)": sor(A, b, x0, omega=1.5, tol_ratio=tol_ratio, max_iterations=max_iterations),
        "Steepest Descent": steepest_descent(A, b, x0, tol_ratio, max_iterations),
        "Conjugate Gradient": conjugate_gradient(A, b, x0, tol_ratio, max_iterations),
    }

    print(f"Laplace equation linear system: k={k}, unknowns={(k - 1) ** 2}")
    print_results(results)
    print_solutions(results)


if __name__ == "__main__":
    main()
