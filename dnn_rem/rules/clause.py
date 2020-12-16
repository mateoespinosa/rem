from dnn_rem.logic_manipulator.delete_redundant_terms import \
    remove_redundant_terms


class ConjunctiveClause(object):
    """
    Represent conjunctive clause. All terms in clause are ANDed together.
    Immutable and Hashable.

    Each conjunctive clause of terms has its own confidence value

    The rank_score refers to the hill-climbing score associated with each
    clause. By default, each clause will have a score of 1.
    """
    def __init__(self, terms=None, confidence=1, score=1):
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
        """
        Returns the conjunctive union of this clause with another clause.

        :param ConjunctiveClause other: The other ConjuctiveClause we want to
            union this clause with.

        :returns ConjunctiveClause: The result of ANDing both this clause and
            the provided clause.
        """
        if not isinstance(other, ConjuctiveClause):
            raise ValueError(
                "We only support && operator with a ConjuctiveClause "
                "if both arguments are ConjuctiveClause. However, one of the "
                f"provided arguments is of type {type(other).__name__}"
            )
        # Return new conjunctive clause that has all terms from both
        return ConjunctiveClause(
            terms=self.terms.union(other.terms),
            # TODO change this vvv ? see when called? its not right
            confidence=(self.confidence * other.confidence),
        )

    def __and__(self, other):
        """
        Bitwise 'and' operator.

        :param ConjuctiveClause other: The other clause which we want to
            perform a conjunctive union with.

        :returns ConjuctiveClause: The result of the 'and' operator between
            the given clauses.
        """
        return self.union(other)

    def evaluate(self, data):

        """
        Evaluate clause with data Dict[Neuron, float]
        """
        for term in self.terms:
            if not term.apply(data[term.neuron]):
                return False

        # All conditions in the clause are satisfied
        return True
