from tqdm import tqdm
import torch
import string

class Tokenizer:

    def __init__(self, max_seq_len):
        self.idcnt = 0
        self.str2id = {}
        self.id2str = {}
        self.max_seq_len = max_seq_len
        self.unk_tokens = []

        self.unary_operators = ['sin', 'cos']
        self.binary_operators = ['+', '-', '*', '/', '^']

        def add_token(token):
            self.str2id[token] = self.idcnt
            self.id2str[self.idcnt] = token
            self.idcnt += 1

        for num in range(20000):
            _ = add_token(str(num))

        _ = add_token('exp')
        for var in string.ascii_lowercase:
            _ = add_token(var)

        for token in ['<BOS>', '<TAR>', '<PAD>', '<EOS>', '<UNK>']:
            _ = add_token(token)

        for op in self.unary_operators + self.binary_operators:
            _ = add_token(op)

    def token2id(self, token):
        if token not in self.str2id:
            #raise Exception(f'token {token} not in dictionary')
            self.unk_tokens.append(token)
            return self.str2id['<UNK>']
            
        return self.str2id[token]

    def parse(self, sample, buffer):
        # var or num
        if len(sample) == 1:
            buffer.append(sample[0])
            return

        for op in self.binary_operators:
            level = 0
            for i in range(len(sample)):
                if sample[i] == '(':
                    level += 1
                elif sample[i] == ')':
                    level -= 1

                if level < 0:
                    raise Exception(f'Invalid expression: {sample}')

                # found an operator not in parenthesis
                if level == 0 and sample[i] == op:
                    buffer.append(op)
                    # Special case: sin^4(...) -> sin(...) ^ 4
                    if op == '^' and sample[i - 1] in self.unary_operators:
                        self.parse(sample[0:1] + sample[3:], buffer)
                        buffer.append(sample[2])
                    else:
                        self.parse(sample[0:i], buffer)
                        self.parse(sample[i+1:], buffer)
                    return

        if sample[0] in self.unary_operators:
            # sin(...)
            buffer.append(sample[0])
            level = 0
            for i in range(1, len(sample) - 1):
                if sample[i] == '(':
                    level += 1
                elif sample[i] == ')':
                    level -= 1
                assert level > 0

            if sample[-1] != ')':
                raise Exception(f'Invalid expression: {sample}')

            # parse ...
            self.parse(sample[2:-1], buffer)
                
            return

        # (...)
        self.parse(sample[1:-1], buffer)
        return

    def prefix2infix(self, prefix):
        stack = []
        for i in prefix[::-1]:
            if i in self.unary_operators:
                assert len(stack) >= 1
                (operand, op_id) = stack.pop()
                stack.append((f'{i}({operand})', 5))

            elif i in self.binary_operators:
                assert len(stack) >= 2
                op_id = self.binary_operators.index(i)
                (operand1, op_id1) = stack.pop()
                (operand2, op_id2) = stack.pop()
                if op_id1 < op_id:
                    operand1 = '(' + operand1 + ')'
                if op_id2 < op_id or (i == '^' and not operand2.isdigit()):
                    operand2 = '(' + operand2 + ')'

                if i == '^':
                    if operand1[0:3] in self.unary_operators and operand2.isdigit():
                        stack.append((f'{operand1[0:3]}^{operand2}{operand1[3:]}', op_id))
                        continue

                ok = False
                if i == '*':
                    if operand1.isdigit() and (operand2[0].isalpha() or operand2[0] == '('):
                        ok = True
                    elif operand1.isalpha() and operand2[0] == '(':
                        ok = True
                if ok:
                    stack.append((f'{operand1}{operand2}', op_id))
                else:
                    stack.append((f'{operand1}{i}{operand2}', op_id))
            else:
                stack.append((i, 6))

        assert len(stack) == 1
        return stack[0][0]

    def _tokenize(self, sample):
        tokens = []
        curidx = 0
        # split all functions
        while curidx < len(sample):
            endidx = curidx
            if sample[curidx].isalpha():
                while endidx + 1 < len(sample) and sample[endidx + 1].isalpha():
                    endidx += 1
            if sample[curidx].isdigit():
                while endidx + 1 < len(sample) and sample[endidx + 1].isdigit():
                    endidx += 1

            cur_token = sample[curidx:endidx + 1]
            if len(tokens) > 0:
                if tokens[-1].isdigit() and (cur_token.isalpha() or cur_token == '('):
                    if len(tokens) > 1 and tokens[-2] == '^':
                        pass
                    else:
                        tokens.append('*')
                elif tokens[-1].isalpha() and tokens[-1] not in self.unary_operators and (cur_token == '('):
                    tokens.append('*')
            tokens.append(cur_token)
            curidx = endidx + 1

        prefix_tokens = []
        self.parse(tokens, prefix_tokens)


        infix = self.prefix2infix(prefix_tokens)
        assert(infix == sample)

        prefix_tokens_id = []
        for token in prefix_tokens:
            prefix_tokens_id.append(self.token2id(token))

        return prefix_tokens_id

    def tokenize(self, sample, test=False):
        if test:
            numerator = sample[2:-4]
            denominator = sample[-1]

            return self._tokenize(numerator), self._tokenize(denominator), None
        else:
            if sample.count('=') != 1:
                raise Exception(f'Invalid expression: {sample}')

            function, derivative = sample.split('=')
            numerator = function[2:-4]
            denominator = function[-1]

            return self._tokenize(numerator), self._tokenize(denominator), self._tokenize(derivative)

    def _preprocess(self, sample, test=False):
        target_token = self.token2id('<TAR>')
        numerator, denominator, derivative = self.tokenize(sample, test=test)

        if len(denominator) != 1:
            raise Exception(f'Invalid expression (too many =): {sample}')

        for i in range(len(numerator)):
            if numerator[i] == denominator[0]:
                numerator[i] = target_token

        if not test:
            for i in range(len(derivative)):
                if derivative[i] == denominator[0]:
                    derivative[i] = target_token

        X = numerator + [self.token2id('<EOS>')]

        if len(X) < self.max_seq_len:
            X += [self.token2id('<PAD>')] * (self.max_seq_len - len(X))

        if not test:
            y = derivative + [self.token2id('<EOS>')]
            if len(y) < self.max_seq_len:
                y += [self.token2id('<PAD>')] * (self.max_seq_len - len(y))

        if not test:
            return (torch.tensor(X), torch.tensor(y), torch.tensor(denominator[0]))
        else:
            return (torch.tensor(X), torch.tensor(denominator[0]))

    def preprocess(self, samples, test=False):
        X, denominators = [], []
        if not test:
            y = []
        for sample in tqdm(samples):
            if not test:
                _X, _y, _denom = self._preprocess(sample, test)
                X.append(_X)
                y.append(_y)
                denominators.append(_denom)
            else:
                _X, _denom = self._preprocess(sample, test)
                X.append(_X)
                denominators.append(_denom)

        if not test:
            return torch.stack(X), torch.stack(y), torch.stack(denominators)
        else:
            return torch.stack(X), torch.stack(denominators)
 
    def _postprocess(self, sample, denominator, to_infix=False):
        target_token = self.token2id('<TAR>')
        for i in range(len(sample)):
            if sample[i] == target_token:
                sample[i] = denominator

        words = [self.id2str[idx] for idx in sample]
        end_idx = words.index('<EOS>') if '<EOS>' in words else len(sample)
        words = words[:end_idx]

        if to_infix:
            return self.prefix2infix(words)
        else:
            return ''.join(words)

    def postprocess(self, samples, denominators, to_infix=False):
        result = []
        for (sample, denom) in zip(samples, denominators):
            try:
                res = self._postprocess(sample, denom, to_infix=to_infix)
                result.append(res)
            except Exception as e:
                res = self._postprocess(sample, denom, to_infix=False)
                result.append(f'err: {res} with reason {e}')

        return result

from pytorch_lightning.callbacks import Callback

class Visualize(Callback):
    def __init__(self, sample_data, tokenizer):
        self.data = sample_data
        self.tokenizer = tokenizer
    def visualize(self, pl_module):
        X, y, denom = self.data
        num = min(10, X.shape[0])
        y_hat = pl_module.predict_step((X[:num], denom[:num]), 0)
        print('='*10)
        for i in range(num):
            _d = denom[i]
            _X = self.tokenizer._postprocess(X[i].tolist(), _d.item())
            _y = self.tokenizer._postprocess(y[i].tolist(), _d.item())
            _y_hat = self.tokenizer._postprocess(y_hat[i].tolist(), _d.item())
            print(_X, _y, _y_hat, _d, sep=' ||| ')
        print()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % 500 == 0:
            self.visualize(pl_module)
