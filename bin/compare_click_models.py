#!/usr/bin/env pypy

from __future__ import print_function

import sys
import glob
import six

from clickmodels.bootstrap import bootstrap
from clickmodels.inference import *
from clickmodels.input_reader import InputReader

try:
    from config import *
except:
    from clickmodels.config_sample import *


def perpGain(r1, r2):
    """ Perplexity gain of r2 over r1. """
    return (r1 - r2) / (r1 - 1)


def llGain(r1, r2):
    """ Log-likelihood gain of r2 over r1. """
    return (math.exp(r2 - r1) - 1)


def avg(l):
    """ Average of an iterable l. """
    s = 0
    n = 0
    for x in l:
        s += x
        n += 1
    return float(s) / n if n else 0


TESTED_MODEL_PAIRS = ['UBM', 'EB_UBM', 'UBMvsDBN']

MODEL_CONSTRUCTORS = {
    'DBN': (lambda config: DbnModel((0.9, 0.9, 0.9, 0.9), config=config),
            lambda config: DbnModel((1.0, 0.9, 1.0, 0.9),
                                    ignoreIntents=False, ignoreLayout=False, config=config)),
    'UBMvsDBN': (lambda config: UbmModel(config=config),
                 lambda config: DbnModel((0.9, 0.9, 0.9, 0.9), config=config)),
    'UBM': (lambda config: UbmModel(config=config),
            lambda config: UbmModel(ignoreIntents=False, ignoreLayout=False, config=config)),
    'EB_UBM': (lambda config: UbmModel(config=config),
               lambda config: EbUbmModel(config=config),
               lambda config: EbUbmModel(ignoreIntents=False, ignoreLayout=False,
                                         config=config)),
    'DCM': (lambda config: DcmModel(config=config),
            lambda config: DcmModel(ignoreIntents=False, ignoreLayout=False, config=config)),
}

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {0:s} directory_with_log_files_for_different_days'.format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)
    perplexityGains = dict((m, defaultdict(lambda: [])) for m in TESTED_MODEL_PAIRS)
    perplexityGainsPos = [dict((m, defaultdict(lambda: [])) for m in TESTED_MODEL_PAIRS) for pos in six.range(MAX_DOCS_PER_QUERY)]
    llGains = dict((m, defaultdict(lambda: [])) for m in TESTED_MODEL_PAIRS)
    interestingFiles = sorted(glob.glob(sys.argv[1] + '/*'))
    N = len(interestingFiles) // 2
    for fileNumber in six.range(N):
        trainFile = interestingFiles[2 * fileNumber]
        testFile = interestingFiles[2 * fileNumber + 1]
        readInput = InputReader(MIN_DOCS_PER_QUERY, MAX_DOCS_PER_QUERY,
                                EXTENDED_LOG_FORMAT, SERP_SIZE,
                                TRAIN_FOR_METRIC,
                                discard_no_clicks=True)
        trainSessions = readInput(open(trainFile))
        testSessions = readInput(open(testFile))
        config = {
            'MAX_QUERY_ID': readInput.current_query_id + 1,
            'MAX_ITERATIONS': MAX_ITERATIONS,
            'DEBUG': DEBUG,
            'PRETTY_LOG': PRETTY_LOG,
            'MAX_DOCS_PER_QUERY': MAX_DOCS_PER_QUERY,
            'SERP_SIZE': SERP_SIZE,
            'TRANSFORM_LOG': TRANSFORM_LOG,
            'QUERY_INDEPENDENT_PAGER': QUERY_INDEPENDENT_PAGER,
            'DEFAULT_REL': DEFAULT_REL
        }
        for modelName in TESTED_MODEL_PAIRS:
            res = []
            models = MODEL_CONSTRUCTORS[modelName]
            for idx, model in enumerate(models):
                m = model(config)
                m.train(trainSessions)
                currentResult = m.test(testSessions, reportPositionPerplexity=True)
                res.append(currentResult)
                print(float(fileNumber) / N, modelName, idx, currentResult, file=sys.stderr)
                del m
            for i in six.range(len(models)):
                for j in six.range(i + 1, len(models)):
                    perplexityGains[modelName][(i, j)].append(perpGain(res[i][1], res[j][1]))
                    llGains[modelName][(i, j)].append(llGain(res[i][0], res[j][0]))
                    for pos in six.range(MAX_DOCS_PER_QUERY):
                        perplexityGainsPos[pos][modelName][(i, j)].append(perpGain(res[i][2][pos], res[j][2][pos]))

    for t in ['ll', 'perplexity']:
        print(t.upper())
        for m in TESTED_MODEL_PAIRS:
            gainsDict = locals()[t + 'Gains'][m]
            for k, gains in six.iteritems(gainsDict):
                print(m, k, gains, bootstrap(gains)[1])
                if t == 'perplexity':
                    print(m, 'POSITION PERPLEXITY GAINS:', k, [[f(perplexityGainsPos[pos][m][k]) for f in [avg, lambda l: bootstrap(l)[1]]] for pos in six.range(MAX_DOCS_PER_QUERY)])

