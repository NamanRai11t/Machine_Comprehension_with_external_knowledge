#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import copy

class conceptnet_labellist():
    def __init__(self):
        self.conceptnet_types = self.def_conceptnet_types()
        self.symmetric_relations = self.def_symmetric_relations()
        
    def __call__(self, coref=False):
        return self.make_fulllist(self.conceptnet_types, self.symmetric_relations, coref)
    
    def def_conceptnet_types(self):
        conceptnet_types = ["Antonym","AtLocation","CapableOf","Causes","CausesDesire","CreatedBy","DefinedAs","DerivedFrom","Desires","DistinctFrom","Entails","EtymologicallyRelatedTo","ExternalURL","FormOf","HasA","HasContext","HasFirstSubevent","HasLastSubevent","HasPrerequisite","HasProperty","HasSubevent","InstanceOf","IsA","LocatedNear","MadeOf","MannerOf","MotivatedByGoal","NotCapableOf","NotDesires","NotHasProperty","NotUsedFor","ObstructedBy","PartOf","ReceivesAction","RelatedTo","SimilarTo","SymbolOf","Synonym","UsedFor","dbpedia/capital","dbpedia/field","dbpedia/genre","dbpedia/genus","dbpedia/influencedBy","dbpedia/knownFor","dbpedia/language","dbpedia/leader","dbpedia/occupation","dbpedia/product"]
        return conceptnet_types

    def def_symmetric_relations(self):
        symmetric_relations = ['RelatedTo', 'SimilarTo', 'EtymologicallyRelatedTo', 'Synonym', 'Antonym', 'DistinctFrom'] # from https://github.com/commonsense/conceptnet5/blob/master/conceptnet5/relations.py
        symmetric_relations += ['LocatedNear'] # from https://github.com/commonsense/conceptnet5/wiki/Relations
        return symmetric_relations
        
    def make_fulllist(self, raw_conceptnet, raw_symmetric, coref=False):
        conceptnet_types = copy.copy(raw_conceptnet)
        symmetric_relations = copy.copy(raw_symmetric)
        # make reverse relation labels
        reverse_relations = []
        for relation in conceptnet_types:
            if not relation in symmetric_relations:
                reverse_relations.append(relation + "_R")
        conceptnet_types.extend(reverse_relations)
        if coref:
            conceptnet_types.append("coref")
        return conceptnet_types
    
if __name__ == '__main__':
    make_labellist = conceptnet_labellist()
    print(make_labellist())
