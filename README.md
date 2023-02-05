# Transformer Derivative Calculator

A transformer-based derivative calculation model. Support plus (`+`), minus (`-`), multiply (`*`), division (`/`), pow (`^`), parenthesis (`(`, `)`), sine function (`sin`), cosine function (`cos`) and exponential (`exp`).

Sample input / output:

* `d(7cos^4(8e^2))/de` -> `448e*sin(8e^2)*cos^3(8e^2)`
* `d(5e^9+15e^7)/de` -> `45e^8+105e^6`
* `d(5z^4+16z^11+4z^4+18z^6)/dz` -> `20z^3+176z^10+16z^3+108z^5`

## Prerequisite

*Requires `python_version=">=3.6, <3.7"`*

## Method

### Preprocess / Postprocess

Instead of using the data directly, note that each expression can be represented by a tree, and the given expression is called infix expression. I normalized the sequence and converted it into prefix expression to remove the parenthesis. After prediction, I converted the predicted prefix sequence back to infix sequence.

1. Split train data into three parts: numerator, denominator, derivative.
2. Normalize each expression and convert them from infix expression to prefix expression. Normalize will do the following:
	* Add `*` back to expression, including `3a -> 3*a`, `3( -> 3*(`, `a( -> a*(`
	* For `sin` and `cos`, move `^`: `sin^5(...) -> sin(...)^5`
3. Tokenize the expressions. Our dictionary contains the following: 
	* numbers: `0-19999`, `exp`
	* alphabets: `a-z` (assuming all denominators is a single alphabets)
	* unary operators: `sin`, `cos`
	* binary operators: `+`, `-`, `*`, `/`, `^`
	* begin of sequence: `<BOS>`
	* end of sequence: `<EOS>`
	* padding: `<PAD>`
	* target: `<TAR>`
	* unknown: `<UNK>`
4. Replace denominators in numerator to target token `<TAR>`
5. Insert `<EOS>` to the back of input (numerator) and output (derivative), then pad to a predefined length `MAX_SEQUENCE_LENGTH`.

### Model

I built a sequence-to-sequence model, utilizing the encoder and decoder in the transformer architecture. I used a weighted cross-entropy loss, given `sin` and `cos` double weight than all other tokens. This is because I found out that the model made a lot mistakes confusing `sin` and `cos`. For the training part I used teacher forcing to improve both the training speed and result.

The resulting model is a 3.1M parameter model, scoring over 98.05% on validation dataset (I split 20% of train data as validation data). Training data provided upon request.
