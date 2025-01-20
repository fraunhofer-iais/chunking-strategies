from src.metrics.reader_metrics import ReaderMetrics


def test_normalize_answer():
    sentence_to_normalize = 'Hi!!! This is a test for normalization, which is    basically cleaning texts.'
    evaluate = ReaderMetrics()
    predicted_sentence = evaluate.normalize_answer(sentence_to_normalize)
    expected_sentence = 'hi this is test for normalization which is basically cleaning texts'
    assert expected_sentence == predicted_sentence, "Correctly normalized"


def test_get_tokens():
    sentence_to_normalize = 'Tesla!!! Digital Key allows you to use a compatible smartphone to lock, unlock and    start the vehicle.'
    evaluate = ReaderMetrics()
    predicted_token = evaluate.get_tokens(sentence_to_normalize)
    expected_token = ['tesla', 'digital', 'key', 'allows', 'you', 'to', 'use', 'compatible', 'smartphone', 'to', 'lock',
                      'unlock', 'and', 'start', 'vehicle']
    assert expected_token == predicted_token, "Correctly tokenized"


def test_compute_exact():
    gold_sentence = 'tesla digital key allows you to use compatible smartphone to lock unlock and start vehicle'
    pred_sentence = 'tesla digital key allows you to use compatible smartphone to lock unlock and start vehicle'
    evaluate = ReaderMetrics()
    exact_match = evaluate.compute_exact(gold_sentence, pred_sentence)
    assert exact_match == 1, "Sentences are exactly same"

    pred_wrong_sentence = 'tesla digial key'
    exact_match = evaluate.compute_exact(gold_sentence, pred_wrong_sentence)
    assert exact_match == 0, 'Sentences donot match'


def test_compute_f1_precision_recall():
    # test for same 'common words' in prediction and gold spans
    gold_tokens = 'tesla digital key allows you to use compatible smartphone to lock unlock and start vehicle'
    pred_tokens = 'tesla digital key allows you to use compatible smartphone to lock unlock and start vehicle'
    evaluate = ReaderMetrics()
    predicted_f1, predicted_precision, predicted_recall = evaluate.compute_f1_precision_recall(gold_tokens, pred_tokens)
    assert predicted_f1 == 1, 'Correct f1 score'
    assert predicted_precision == 1, 'Correct precision'
    assert predicted_recall == 1, 'Correct recall'

    # test for same 'common words' (9 common words) in prediction and gold spans with gold span being longer
    gold_tokens = 'tesla digital key allows you to use compatible smartphone to lock unlock and start vehicle'
    pred_tokens = 'tesla digital key allows you to use compatible smartphone'
    predicted_f1, predicted_precision, predicted_recall = evaluate.compute_f1_precision_recall(gold_tokens, pred_tokens)
    assert predicted_f1 == 0.7499999999999999, 'Correct f1 score'
    assert predicted_precision == 1, 'Correct precision'
    assert predicted_recall == 0.6, 'Correct recall'

    # test for same 'common words' (9 common words) in prediction and gold spans with gold span being shorter
    pred_tokens = 'tesla digital key allows you to use compatible smartphone to lock unlock and start vehicle'
    gold_tokens = 'tesla digital key allows you to use compatible smartphone'
    predicted_f1, predicted_precision, predicted_recall = evaluate.compute_f1_precision_recall(gold_tokens, pred_tokens)
    assert predicted_f1 == 0.7499999999999999, 'Correct f1 score'
    assert predicted_precision == 0.6, 'Correct precision'
    assert predicted_recall == 1, 'Correct recall'

    # test for no 'common words' in prediction and gold spans
    gold_tokens = 'tesla digital key allows you to use compatible smartphone to lock unlock and start vehicle'
    pred_tokens = 'it is high tech security system'
    predicted_f1, predicted_precision, predicted_recall = evaluate.compute_f1_precision_recall(gold_tokens, pred_tokens)
    assert predicted_f1 == 0, 'Correct f1 score'
    assert predicted_precision == 0, 'Correct precision'
    assert predicted_recall == 0, 'Correct recall'

    # test for empty prediction spans
    gold_tokens = 'tesla digital key allows you to use compatible smartphone to lock unlock and start vehicle'
    pred_tokens = ''
    predicted_f1, predicted_precision, predicted_recall = evaluate.compute_f1_precision_recall(gold_tokens, pred_tokens)
    assert predicted_f1 == 0, 'Correct f1 score'
    assert predicted_precision == 0, 'Correct precision'
    assert predicted_recall == 0, 'Correct recall'
