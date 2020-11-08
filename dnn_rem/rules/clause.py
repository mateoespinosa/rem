from . import DELETE_REDUNDANT_TERMS_FLAG
from dnn_rem.logic_manipulator.delete_redundant_terms import \
    remove_redundant_terms


class ConjunctiveClause(object):
    """
    Represent conjunctive clause. All terms in clause are ANDed together.
    Immutable and Hashable.

    Each conjunctive clause of terms has its own confidence value
    """
    __slots__ = [
        'terms',
        'confidence',
    ]

    def __init__(self, terms=None, confidence=1):
        terms = terms or set()

        if DELETE_REDUNDANT_TERMS_FLAG:
            terms = remove_redundant_terms(terms)

        super(ConjunctiveClause, self).__setattr__('terms', terms)
        super(ConjunctiveClause, self).__setattr__('confidence', confidence)

    def __str__(self):
        terms_str = [str(term) for term in self.terms]
        return f"{self.confidence}[{' AND '.join(terms_str)}]"

    def __eq__(self, other):
        return (
            isinstance(other, ConjunctiveClause) and
            (self.terms == other.terms)
        )

    def __hash__(self):
        return hash(tuple(self.terms))

    def get_terms(self):
        return self.terms

    def get_confidence(self):
        return self.confidence

    def union(self, other):
        # Return new conjunctive clause that has all terms from both
        terms = self.terms.union(other.get_terms())
        # TODO change this vvv ? see when called? its not right
        confidence = self.confidence * other.get_confidence()

        return ConjunctiveClause(terms=terms, confidence=confidence)

    def evaluate(self, data):
        """
        Evaluate clause with data Dict[Neuron, float]
        """
        for term in self.terms:
            if not term.apply(data[term.get_neuron()]):
                return False

        # All conditions in the clause are satisfied
        return True
