from dnn_rem.logic_manipulator.delete_redundant_terms import \
    remove_redundant_terms


class ConjunctiveClause(object):
    """
    Represent conjunctive clause. All terms in clause are ANDed together.
    Immutable and Hashable.

    Each conjunctive clause of terms has its own confidence value

    The rank_score refers to the hill-climbing score associated with each
    clause.
    """
    def __init__(self, terms=None, confidence=1, score=0):
        self.terms = remove_redundant_terms(terms or set())
        self.confidence = confidence
        self.score = score

    def __str__(self):
        terms_str = [str(term) for term in sorted(self.terms, key=str)]
        return f"{self.confidence}[{' AND '.join(terms_str)}]"

    def __eq__(self, other):
        return (
            isinstance(other, ConjunctiveClause) and
            (self.terms == other.terms)
        )

    def __hash__(self):
        return hash(tuple(self.terms))

    def union(self, other):
        # Return new conjunctive clause that has all terms from both
        terms = self.terms.union(other.terms)
        # TODO change this vvv ? see when called? its not right
        confidence = self.confidence * other.confidence

        return ConjunctiveClause(terms=terms, confidence=confidence)

    def evaluate(self, data):
        """
        Evaluate clause with data Dict[Neuron, float]
        """
        for term in self.terms:
            if not term.apply(data[term.neuron]):
                return False

        # All conditions in the clause are satisfied
        return True
