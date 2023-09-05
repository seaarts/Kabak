from numpy.typing import ArrayLike
from ortools.linear_solver import pywraplp


def _set_variables(solver, d: ArrayLike, kind: str = "fractional"):
    """Attach variables to ortools.pwraplp.Solver"""

    x = {}

    for i, multiplicity in enumerate(d):
        if multiplicity == float("inf"):
            multiplicity = solver.infinity()

        if kind == "fractional":
            x[i] = solver.NumVar(0, multiplicity, f"x[{i}]")

        if kind == "integral":
            x[i] = solver.IntVar(0, multiplicity, f"x[{i}]")

    return x


def _set_covering_constraints(solver, x, A, b):
    """Attach covering constrainats to ortools.pywraplp.Solver"""

    n_vars = A.shape[1]

    covering_constraints = {}

    for j, bound in enumerate(b):
        # Each user has a list of constraints in dict covering_constraints
        covering_constraints[f"{j}"] = []

        constraint_expr = [A[j][i] * x[i] for i in range(n_vars)]

        covering_constraints[f"{j}"].append(solver.Add(sum(constraint_expr) >= bound))

    return covering_constraints


def _set_packing_constraints(solver, x, B, f):
    """Attach packing constrainats to ortools.pywraplp.Solver"""

    n_vars = B.shape[1]

    for j, bound in enumerate(f):
        constraint_expr = [B[j][i] * x[i] for i in range(n_vars)]
        solver.Add(sum(constraint_expr) <= bound)

    # return solver, x


def _set_objective(solver, x, c, kind: str = "minimimze"):
    """Set the objective of ortools.pywraplp.Solver."""
    for i, cost in enumerate(c):
        solver.Objective().SetCoefficient(x[i], cost)

    if kind == "minimize":
        solver.Objective().SetMinimization()
    elif kind == "maximize":
        solver.Objective().SetMaximization()

    else:
        raise ValueError(f"Invalid value for {kind = }. Try `minimize` or `maximize`.")

    # return solver, x


def _make_linear_program(
    c: ArrayLike,
    A: ArrayLike,
    b: ArrayLike,
    B: ArrayLike,
    f: ArrayLike,
    d: ArrayLike,
    minimize: bool = True,
    integral: bool = False,
    solver_type: int = pywraplp.Solver.CLP_LINEAR_PROGRAMMING,
):
    """Returns a Gurobi model for the LP / IP, as well as pointers to the
    variables x, covering constraints, and packing constraints.

    Parameters
    ----------
    c : ArrayLike
        Costs of each item.
    A : ArrayLike
        Matrix of covering constraints.
    b : ArrayLike
        Vector of covering demands.
    B : ArrayLike
        Matrix of packing constraints.
    f : ArrayLike
        Vector of packing budgets.
    d : ArrayLike
        Multiplicity constraints
    minimize: bool
        Whether to minimize the objective, if ``False`` it is maximized.
    integral : bool
        Whether variables are integral. If ``False`` fractional solutions
        are returned.
    solver_type : int
        Spefifies which OR-tools solver to use; see the
        `or-tools LP reference
        <https://developers.google.com/optimization/lp/lp_advanced#families_of_lp_algorithms>`_
        for options. Default is ``0``.
    """
    if integral:
        var_kind = "integral"
    else:
        var_kind = "fractional"

    if minimize:
        opt_kind = "minimize"
    else:
        opt_kind = "maximize"

    solver = pywraplp.Solver("KabakLP", solver_type)

    x = _set_variables(solver, d, kind=var_kind)

    covering_constraints, packing_constraints = None, None

    if not A is None and not b is None:
        covering_constraints = _set_covering_constraints(solver, x, A, b)

    if not B is None and not f is None:
        packing_constraints = _set_packing_constraints(solver, x, B, f)

    _set_objective(solver, x, c, kind=opt_kind)

    return solver, x, covering_constraints, packing_constraints


def linear_program_ortools(
    c: ArrayLike,
    A: ArrayLike,
    b: ArrayLike,
    B: ArrayLike,
    f: ArrayLike,
    d: ArrayLike,
    minimize: bool = True,
    integral: bool = False,
    solver_type: str = "GLOP",
) -> dict:
    """Solve linear programming relaxation of the problem with Google OR-tools.

    Parameters
    ----------
    c : ArrayLike
        Costs of each item.
    A : ArrayLike
        Matrix of covering constraints.
    b : ArrayLike
        Vector of covering demands.
    B : ArrayLike
        Matrix of packing constraints.
    f : ArrayLike
        Vector of packing budgets.
    d : ArrayLike
        Multiplicity constraints
    minimize: bool
        Whether to minimize the objective, if ``False`` it is maximized.
    integral : bool
        Whether variables are integral. If ``False`` fractional solutions
        are returned.
    solver_type : str
        Spefifies which OR-tools solver to use; see the
        `or-tools LP reference
        <https://developers.google.com/optimization/lp/lp_advanced#families_of_lp_algorithms>`_
        for options. Default is ``"GLOP"``.
    """

    solver, x, _, _ = _make_linear_program(
        c, A, b, B, f, d, minimize, integral, solver_type
    )

    status = solver.Solve()

    val, sol = -1, []

    if status == pywraplp.Solver.OPTIMAL:
        val = solver.Objective().Value()
        sol = [var.solution_value() for _, var in x.items()]

    return {"val": val, "sol": sol, "status": status}
