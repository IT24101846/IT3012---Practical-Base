"""A small propositional-logic knowledge base for grid-agent decisions."""


class KnowledgeBase:
    """Stores facts and Horn-clause rules, and derives facts by forward chaining."""

    def __init__(self):
        self.facts = set()
        self.rules = []

    def tell_fact(self, fact_string):
        """Add one fact; duplicate facts are ignored by the set."""
        self.facts.add(fact_string)

    def tell_rule(self, premise_list, conclusion_string):
        """Add a rule of the form premises => conclusion."""
        self.rules.append((list(premise_list), conclusion_string))

    def clear_facts(self):
        """Remove all currently known and inferred facts."""
        self.facts.clear()

    def forward_chain(self):
        """Infer facts until no rule can add a new conclusion."""
        new_facts_added = True

        while new_facts_added:
            new_facts_added = False

            for premises, conclusion in self.rules:
                if conclusion not in self.facts and all(premise in self.facts for premise in premises):
                    self.facts.add(conclusion)
                    new_facts_added = True
