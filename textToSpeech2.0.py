from google.cloud import speech_v1 as speech
from google.oauth2 import service_account
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support

creds = service_account.Credentials.from_service_account_file("/home/frank/Downloads/auth.json")

client = speech.SpeechClient(credentials=creds)

with open("audio/Homonyms/dear_dear_vincent.wav", 'rb') as audio_file:
    content = audio_file.read()

ground_truth = "She’s such a dear friend. The deer ate all the lettuce in the garden."


audio = speech.types.RecognitionAudio(content=content)

config = speech.types.RecognitionConfig(
    encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=44100,
    language_code='en-US')

timeBefore = datetime.now()
response = client.recognize(config, audio)
RTF = datetime.now() - timeBefore


def wer(ref, hyp, debug=False):
    """
    Calculate the word error rate
    :param ref: The transcript of the audio
    :param hyp: The recognized string from the algorithm
    :param debug: Log debug information. Default false
    :return: The word error rate
    :return: The word correct rate
    """
    r = ref.split()
    h = hyp.split()
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    DEL_PENALTY = 1  # Tact
    INS_PENALTY = 1  # Tact
    SUB_PENALTY = 1  # Tact
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i] + "\t" + "****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("Ncor " + str(numCor))
        print("Nsub " + str(numSub))
        print("Ndel " + str(numDel))
        print("Nins " + str(numIns))
        return (numSub + numDel + numIns) / (float)(len(r))

    wer = round((numSub + numDel + numIns) / (float)(len(r)), 3)
    wcr = round((numCor / len(r)), 3)
    return wer, wcr


for i, result in enumerate(response.results):
    alternative = result.alternatives[0]

    ground_truth = ground_truth.lower()
    ground_truth = ground_truth.replace('.', '')
    ground_truth = ground_truth.replace(',', '')

    transcript = alternative.transcript.lower()

    # Convert string to char array
    true = list(ground_truth)
    estimate = list(transcript)

    size_of_true = len(true)
    size_of_estimate = len(estimate)

    list_of_truth = ground_truth.split()
    list_of_transcript = transcript.split()

    size_of_truth = len(ground_truth.split()) - 1
    size_of_transcript = len(transcript.split()) - 1

    amount_of_goodstuff = 0

    size_diff = abs(size_of_true - size_of_estimate)

    # Make both lists the same size by adding empty strings
    if size_diff:
        if len(true) > len(estimate):
            for i in range(size_diff):
                estimate.append('')
        else:
            for i in range(size_diff):
                true.append('')

    precision_micro, recall_micro, f_score_micro, true_sum_micro = precision_recall_fscore_support(true, estimate,
                                                                                                   average='micro',
                                                                                                   zero_division=0)

    precision_macro, recall_macro, f_score_macro, true_sum_macro = precision_recall_fscore_support(true, estimate,
                                                                                                   average='macro',
                                                                                                   zero_division=0)
    wer, wcr = wer(ground_truth, transcript, False)

    print('-' * 20)

    print('precision_micro = ' + str(round(precision_micro, 3)))
    print('precision_macro = ' + str(round(precision_macro, 3)))
    print()
    print('recall_micro = ' + str(round(recall_micro, 3)))
    print('recall_macro = ' + str(round(recall_macro, 3)))
    print()
    print('f_score_micro = ' + str(round(f_score_micro, 3)))
    print('f_score_macro = ' + str(round(f_score_macro, 3)))
    print()
    print('WER = ' + str(wer))
    print('WCR = ' + str(wcr))
    print('RTF = ' + str(RTF))
    print()
    print(u'True: {}'.format(ground_truth))
    print(u'Estimate: {}'.format(transcript))


def wer(ref, hyp, debug=False):
    """
    Calculate the word error rate
    :param ref: The transcript of the audio
    :param hyp: The recognized string from the algorithm
    :param debug: Log debug information. Default false
    :return: The word error rate
    :return: The word correct rate
    """
    r = ref.split()
    h = hyp.split()
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    DEL_PENALTY = 1  # Tact
    INS_PENALTY = 1  # Tact
    SUB_PENALTY = 1  # Tact
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i] + "\t" + "****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("Ncor " + str(numCor))
        print("Nsub " + str(numSub))
        print("Ndel " + str(numDel))
        print("Nins " + str(numIns))
        return (numSub + numDel + numIns) / (float)(len(r))

    wer = round((numSub + numDel + numIns) / (float)(len(r)), 3)
    wcr = round((numCor / len(r)), 3)
    return wer, wcr
