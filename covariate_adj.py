from missforest.missforest import MissForest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from graphviz import Digraph
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

class Vertex:
    """
    Vertex class
    """

    def __init__(self, name: str):
        """
        Constructor for the Vertex class
        """
        self.name = name
        self.parents = set() # set consisting of Vertex objects that are parents of this vertex
        self.children = set() # set consisting of Vertex objects that are children of this vertex

class CausalDAG:
    """
    DAG class
    """

    def __init__(self, vertex_names: list[str], edges: list[(str, str)]) -> None:
        """
        Constructor for the causal DAG class
        """

        self.vertices = {v: Vertex(v) for v in vertex_names} # dictionary mapping vertex names to Vertex objects
        self.edges = [] # list of tuples corresponding to edges in the DAG

        # loop over and initialize all vertices to have parent-child relations specified by the edges
        for parent_name, child_name in edges:
            self.edges.append((parent_name, child_name))
            # get the corresponding vertex objects
            parent_vertex = self.vertices.get(parent_name)
            child_vertex = self.vertices.get(child_name)
            # add to the parent/child sets
            parent_vertex.children.add(child_vertex)
            child_vertex.parents.add(parent_vertex)

    def get_parents(self, vertex_name: str) -> list[str]:
        """
        Returns a list of names of the parents
        """
        return [p.name for p in self.vertices[vertex_name].parents]

    def get_children(self, vertex_name: str) -> list[str]:
        """
        Returns a list of names of the children
        """
        return [c.name for c in self.vertices[vertex_name].children]

    def get_neighbors(self, vertex_name: str) -> list[str]:
        """
        Returns a list of names of the neighbors
        """
        return [n.name for n in self.vertices[vertex_name].neighbors]

    def get_descendants(self, vertex_name: str) -> list[str]:
        """
        Returns a list of strings corresponding to descendants of the given vertex.
        Note by convention, the descendants of a vertex include the vertex itself.
        """

        stack = [vertex_name]
        visited = set()

        while len(stack) > 0:

            v_name = stack.pop()
            if v_name in visited:
                continue
            visited.add(v_name)
            stack += self.get_children(v_name)

        return list(visited)


    def d_separated(self, x_name: str, y_name: str, z_names: list[str]) -> bool:
        """
        Check if X _||_ Y | Z using d-separation
        """

        # implement this
        stack = [(x_name, "up")]  # Initialize the stack with the starting vertex X going up
        visited = set() # Set of vertices that have already been visited

        while stack:
            vertex, direction = stack.pop()

            if (vertex, direction) in visited:
                continue  # Skip if the vertex has already been visited

            if vertex == y_name:
                return False  # If Y is reached, X and Y are not d-separated given Z

            visited.add((vertex, direction))

            if direction == "up" and vertex not in z_names:
                for child in self.get_children(vertex):
                    stack.append((child, "down"))
                # If going up and the vertex is not in Z, explore parents
                for parent in self.get_parents(vertex):
                    stack.append((parent, "up"))

            elif direction == "down":
                if vertex not in z_names:
                    for child in self.get_children(vertex):
                        stack.append((child, "down"))
                elif any(descendant in z_names for descendant in self.get_descendants(vertex)):
                    for parent in self.get_parents(vertex):
                        stack.append((parent, "up"))

        return True  # If the loop completes, X and Y are d-separated given Z

    def valid_backdoor_set(self, a_name: str, y_name: str, z_names: list[str]) -> bool:
        """
        Check if Z is a valid backdoor set for computing the effect of A on Y
        """

        # Check the descendant condition
        descendants_a = self.get_descendants(a_name)
        for descendant in descendants_a:
          if descendant in z_names:
            return False  # Z contains descendants of A

        # Create a new CausalDAG object with omitted edges A → C for all C ∈ ChG(A)
        modified_dag_edges = [(parent, child) for parent, child in self.edges if parent != a_name]
        modified_dag = CausalDAG(list(self.vertices.keys()), modified_dag_edges)

        # Check the d-separation condition in the modified DAG
        if not modified_dag.d_separated(a_name, y_name, z_names):
            return False  # A is not d-separated from Y given Z in the modified DAG
        return True  # Z is a valid backdoor set


    def draw(self):
        """
        Method for visualizing the DAG
        """

        dot = Digraph()
        dot.graph_attr["rankdir"] = "LR"

        for v_name in self.vertices:
            dot.node(
                v_name,
                shape="plaintext",
                height=".5",
                width=".5",
            )

        for parent, child in self.edges:
            dot.edge(parent, child, color="blue")

        return dot


def backdoor_adjustment(data: pd.DataFrame, a_name: str, y_name: str, z_names: list[str]) -> float:
    """
    Perform backdoor adjustment for a given treatment A and outcome Y using
    the covariates in Z
    """

    # implement this
    z_names = ["1"] + z_names
    z_formula = " + ".join(z_names)
    regression_formula = f"{y_name} ~ {z_formula} + {a_name}"

    # fit a regression depending on whether Y is binary or not
    if set(data[y_name]) == {0, 1}:
        model = smf.glm(formula=regression_formula, family=sm.families.Binomial(), data=data).fit()
    else:
        model = smf.glm(formula=regression_formula, family=sm.families.Gaussian(), data=data).fit()

    data_a1 = data.copy()
    data_a1[a_name] = 1
    data_a0 = data.copy()
    data_a0[a_name] = 0

    return round(np.mean(model.predict(data_a1) - model.predict(data_a0)), 3)


def compute_confidence_intervals(data: pd.DataFrame, a_name: str, y_name: str, z_names: list[str],
                                 num_bootstraps: int=200, alpha: float=0.05) -> tuple[float, float]:
    """
    Compute confidence intervals for backdoor adjustment via bootstrap

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """

    Ql = alpha / 2
    Qu = 1 - alpha / 2
    estimates = []

    for i in range(num_bootstraps):

        # resample the data with replacement
        data_sampled = data.sample(len(data), replace=True)
        data_sampled.reset_index(drop=True, inplace=True)

        # add estimate from resampled data
        estimates.append(backdoor_adjustment(data_sampled, a_name, y_name, z_names))

    # calculate the quantiles
    quantiles = np.quantile(estimates, q=[Ql, Qu])
    q_low = quantiles[0]
    q_up = quantiles[1]

    return round(q_low, 3), round(q_up, 3)

def findProperCausalPath(G, X, Y):
    """
    Returns all proper causal paths from the treatment X to outcome Y
    """
    result = []
    def dfs(current_vertex, path):
        path.append(current_vertex.name)
        if current_vertex.name == Y:
            result.append(path[:])
        else:
            for child in current_vertex.children:
                dfs(child, path)
        path.pop()  # backtrack

    source_vertex = G.vertices.get(X)
    if source_vertex:
        dfs(source_vertex, [])

    # Change format of path representation
    newPath = []
    for path in result:
        currPath = []
        for i in range(0, len(path) - 1):
            currPath.append((path[i], path[i + 1]))
        newPath.append(currPath)
    return newPath


def properBackdoorGraph(G, paths):
    """
    Returns a directed graph G after removing the first edge of every proper causal paths from X to Y.
    """

    edges_to_remove = [path[0] for path in paths if path]
    modified_edges = []
    for edge in G.edges:
        if edge not in edges_to_remove:
            modified_edges += [edge]
    # Remove duplicate from a list
    modified_edges = list(set(modified_edges))
    return CausalDAG(list(G.vertices.keys()), modified_edges)

def isAncestor(G, X, R_W):
    """
    Returns True if X is an ancestor of any of the variables in list R_W and False otherwise.
    """
    for vertex in R_W:
        stack = [vertex]
        while len(stack) > 0:
            curVertex = stack.pop()
            if curVertex == X:
                return True
            for parent in G.get_parents(curVertex):
                    stack.append(parent)

    return False

def findGXBarBelow(G, X):
    """
    This function returns a directed graph where all outgoing edges from X are deleted.
    """
    # copy a graph G
    modified_dag_edges = [(parent, child) for parent, child in G.edges if parent != X]
    modified_dag = CausalDAG(list(G.vertices.keys()), modified_dag_edges)
    return modified_dag

def findGXBarAbove(G, X):
    """
    This function returns a directed graph where all incoming edges to X are deleted.
    """
    modified_dag_edges = [(parent, child) for parent, child in G.edges if child != X]
    modified_dag = CausalDAG(list(G.vertices.keys()), modified_dag_edges)
    return modified_dag

def descend_pcp(G, X, Y):
    """
    Returns descendants of variables in the proper causal paths from X to Y.
    """
    D_pcp = set()
    properCausalPaths = findProperCausalPath(G, X, Y)
    for path in properCausalPaths:
        for edge in path:
            for vertex in edge:
                descendants = G.get_descendants(vertex)
                for var in descendants:
                    D_pcp.add(var)
    return D_pcp

def valid_MAdj(G, X: str, Y: str, Z: list[str], V):
    """
    V: a list of tuples that contains information on each variable in the DAG with its corresponding missingness
    mechanism. If a variable is fully observed, the second element of the tuple is None.
    Returns a bool value, True if the set Z is a valid m-adjustment set and False otherwise.
    """

    D_pcp = descend_pcp(G, X, Y)
    R_W = []
    for pair in V:
        if pair[1] is not None:
            R_W.append(pair[1])

    # Condition 1: No vertex in Z should be in D_pcp
    if any(vertex in D_pcp for vertex in Z):
        return False

    # Condition 2: Y is d-separated from X given Z and R_W in the proper backdoor graph of G with respect to X and Y.
    properCausalPaths = findProperCausalPath(G, X, Y)
    G_pbd = properBackdoorGraph(G, properCausalPaths)
    if not G_pbd.d_separated(Y, X, Z + R_W):
        return False

    # Condition 3: Y and R_W are d-separated given X in G where all incoming edges from X are deleted.
    GXBarAbove = findGXBarAbove(G, X)
    for R in R_W:
        if not GXBarAbove.d_separated(Y, R, [X]):
            return False

    # Condition 4: If X is an ancestor of any variable in R_W, then it should be d-separated from Y in G where
    # all outgoing edges are deleted.
    if isAncestor(G, X, R_W):
        GXBarBelow = findGXBarBelow(G, X)
        if not GXBarBelow.d_separated(X, Y, []):
            return False

    return True

