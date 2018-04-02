adj_endings_male_hard = ['ый', 'ого', 'ому', 'ым', 'ом', 'ой']
adj_endings_male_soft = ['ий', 'его', 'ему', 'им', 'ем']
adj_endings_female_hard = ['ая', 'ой', 'ую']
adj_endings_female_soft = ['яя', 'ей', 'юю']
adj_endings_neuter_hard = ['ое', 'ого', 'ому', 'ым']
adj_endings_neuter_soft = ['ее', 'его', 'ему', 'им', 'ем']


def is_ending(ending):
    if (ending in adj_endings_female_hard
            or ending in adj_endings_female_soft
            or ending in adj_endings_male_hard
            or ending in adj_endings_male_soft
            or ending in adj_endings_neuter_hard
            or ending in adj_endings_neuter_soft):
        return True
    else:
        return False


def stem(adjective):
    if len(adjective) < 3:
        return adjective
    if is_ending(adjective[-3:]):
        return adjective[:-3]
    if is_ending(adjective[-2:]):
        return adjective[:-2]
    return adjective