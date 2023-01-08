samplers = {
	'Random':                r'Random',
	'StratifiedRandom':      r'Random (s)',
	'AvgImportance':         r'Importance (mean)',
	'MaxImportance':         r'Importance (max)',
	'ZeroImportance':        r'Importance (zero)',
	'StratifiedKMeans20000': r'K-means++',
	'StratifiedFilterCorr':  r'Filter correlation',
}

samplers_short = {
	'Random':                r'R',
	'StratifiedRandom':      r'R (s)',
	'AvgImportance':         r'I ($\mu$)',
	'MaxImportance':         r'I (max)',
	'ZeroImportance':        r'I (0)',
	'StratifiedKMeans20000': r'K++',
	'StratifiedFilterCorr':  r'F',
}

features = {
	'avg_birth_death':           r'Mean birth and death',
	'avg_birth_death_squared':   r'Mean birth and death (squared)',
	'avg_birth_death_inverted' : r'Mean birth and death (log)',
	'std_birth_death':           r'Std birth and death',
	'avg_life':                  r'Mean life',
	'avg_life_squared':          r'Mean life (squared)',
	'avg_half_life':             r'Mean midlife',
	'avg_half_life_squared':     r'Mean midlife (squared)',
}
