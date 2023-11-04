import torch.nn as nn
import random
import torch


graph = {
    0: [(1, "b")],
    1: [(2, "t"), (3, "p")],
    2: [(2, "s"), (4, "x")],
    3: [(3, "t"), (5, "v")],
    4: [(3, "x"), (6, "s")],
    5: [(4, "p"), (6, "v")],
    6: [(-1, "e")],
}


def randomly_traverse_graph(graph):
    curr_node = 0
    sentence = ""
    while True:
        next_states = graph[curr_node]
        if len(next_states) > 1:
            next_state = random.choice(next_states)
        else:
            next_state = next_states[0]
        next_node = next_state[0]
        next_letter = next_state[1]

        sentence += next_letter
        curr_node = next_node
        if curr_node == -1:
            break

    return sentence


def generate_n_samples(graph, num_samples, min_length, max_length):
    samples = set()
    while len(samples) < num_samples:
        sample = randomly_traverse_graph(graph)
        if len(sample) < min_length or len(sample) > max_length:
            continue
        samples.add(sample)
    samples = list(samples)
    return samples


def validate_string(graph, string):
    curr_node = 0
    for letter in string:
        if curr_node == -1:
            print(
                f"{string=} is invalid, reached end of sequence but untraversed data remains"
            )
            return False
        next_states = graph[curr_node]
        is_valid = False
        for state in next_states:
            if letter == state[1]:
                is_valid = True
                curr_node = state[0]
                break
        if not is_valid:
            print(f"{string=} is invalid, expected {next_states=} but got {letter=}")
            return False
    return True


def generate_training_target(sequence):
    alphabet = "btsxpve"
    curr_node = 0
    targets = []
    for letter in sequence:
        # Go to the next node in the graph
        next_state = None
        for option in graph[curr_node]:
            if letter == option[1]:
                next_state = option
                break
        if next_state is None:
            print(f"Error, this sequence is invalid at this {letter=}: {sequence=}")
            break
        # Look at the next node options
        next_node = next_state[0]
        target = [0 for _ in range(len(alphabet))]
        if next_node == -1:
            targets.append(target)
            break
        # One hot encode the next possible
        for option in graph[next_node]:
            for ind, letter in enumerate(alphabet):
                if letter == option[1]:
                    target[ind] = 1
                    break
        curr_node = next_node
        targets.append(target)
    return targets


def convert_string_to_one_hot_sequence(reber_string):
    sequence = []
    alphabet = "btsxpve"
    for letter in reber_string:
        vector = [0 for _ in range(len(alphabet))]
        for ind, elem in enumerate(alphabet):
            if elem == letter:
                vector[ind] = 1
                break
        sequence.append(vector)
    return sequence


def convert_one_hot_sequence_to_string(one_hot_sequence):
    alphabet = "btsxpve"
    reber_string = ""
    for vec in one_hot_sequence:
        reber_string += alphabet[int(vec[0].nonzero())]
    return reber_string


def generate_training_data(
    graph: dict, num_samples: int, min_length: int, max_length: int
) -> list[tuple[torch.tensor]]:
    samples = generate_n_samples(graph, num_samples, min_length, max_length)

    training_data = []
    for sample in samples:
        # convert sample to sequence of one-hot
        input = convert_string_to_one_hot_sequence(sample)
        target = generate_training_target(sample)
        # convert to torch tensors
        input = [torch.tensor(one_hot).reshape(1, 7) for one_hot in input]
        target = [torch.tensor(output).reshape(1, 7) for output in target]
        training_data.append((input, target))
    return training_data


class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.hidden_transform = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_transform = nn.Linear(hidden_size, output_size)

        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.hidden_transform(combined)
        hidden = self.tanh(hidden)
        output = self.output_transform(hidden)
        output = self.sigmoid(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def train_one_example(
    rnn: nn.Module,
    target: list[torch.tensor],
    input_sequence: list[torch.tensor],
    learning_rate: float,
    criterion,
):
    rnn.zero_grad()
    hidden = rnn.initHidden()
    loss = 0
    for i in range(len(input_sequence)):
        output, hidden = rnn(input_sequence[i], hidden)
        loss += (1 / len(input_sequence)) * criterion(output, target[i].float())

    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def train(
    rnn: nn.Module,
    epochs: int,
    training_data: list[tuple[torch.tensor]],
    learning_rate: float,
    criterion,
):
    for _ in range(epochs):
        epoch_loss = 0
        for ind in range(len(training_data)):
            sequence, target = training_data[ind]
            output, loss = train_one_example(
                rnn, target, sequence, learning_rate, criterion
            )
            epoch_loss += loss
        print(epoch_loss)


def eval_one_input(rnn: nn.Module, input: list[torch.tensor]) -> bool:
    hidden = rnn.initHidden()
    for ind, letter in enumerate(input):
        if ind == len(input) - 1:  # the model succeeded
            continue
        prediction, hidden = rnn(letter, hidden)
        _, indices = prediction.sort()
        next_letter = input[ind + 1][0]
        if int(next_letter.nonzero()) not in indices[0][-2:]:
            print(
                f"Network incorrectly predicted {prediction=} at {ind=}, next letter was {next_letter=}"
            )
            return False
    return True


def eval_model(rnn: nn.Module, test_data: list[tuple[torch.tensor]]) -> None:
    rnn.eval()
    num_passed = 0
    for sequence, _ in test_data:
        reber_string = convert_one_hot_sequence_to_string(sequence)
        passed: bool = eval_one_input(rnn, sequence)
        if passed:
            print(f"Network passed on {reber_string=}")
            num_passed += 1
        else:
            print(f"Network failed on {reber_string=}")

    pass_rate = num_passed / len(test_data)
    print(f"Overal pass rate: {pass_rate=}")


def main():
    # Specify model settings
    n_hidden = 4
    input_size = 7
    output_size = 7
    rnn = RNN(input_size, n_hidden, output_size)

    # Specify dataset settings
    num_samples = 400
    min_length = 30
    max_length = 52
    data = generate_training_data(graph, num_samples, min_length, max_length)
    training_data = data[: int(0.8 * num_samples)]
    test_data = data[int(0.8 * num_samples) :]
    # add some random data with shorter strings to test set as well
    test_data = test_data + generate_training_data(graph, 100, 10, 30)

    # Specify training settings
    learning_rate = 1
    epochs = 20
    criterion = nn.BCELoss()
    train(rnn, epochs, training_data, learning_rate, criterion)

    # Evaluate model
    eval_model(rnn, test_data)


if __name__ == "__main__":
    main()
