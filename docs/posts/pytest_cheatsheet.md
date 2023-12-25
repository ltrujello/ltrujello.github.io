---
date: 2023-12-10
---

# Testing Cheatsheet for Python

To successfully manage complex software systems, engineers need to write tests. The reason is because every change 
you make to an existing, working software system is inherently a risk of introducing a bug. 
Unit tests are a very simple line of defense against creating such bugs, and good tests will 
generally run over all possible code paths to relieve developers of the mental load towards making trivial errors. 

<!-- more -->


However, writing tests can sometimes be challenging, because you need to locally run code that is meant to be deployed 
in a cloud environment, and this code usually has logic that manipulates access to external resources and session handlers. Obviously, 
to locally run this code, you need to mock some of the logic. 
Mocking is what usually leads to many woes; sometimes, your test is passing, but later the code breaks anyways, because you 
later find out you didn't write the test properly. Or, something just won't get mocked correctly, and it starts taking up too much time. 

As a python developer, I personally use the pytest, pytest-mock, and unittest testing frameworks. 
Over time I've built up a Pytest cheatsheet so that later I can save my future-self time and remember how to mock and test different types of code objects. 

## Useful Pytest commands

This command allows you to run your tests as you normally would, except it prints the logs of the code being tested. 
Normally, pytest suppresses these logs. You can also control the logging level so you will only see the logs you want to see.

```bash
$ pytest -o log_cli=true --log-cli-level=INFO file.py
```

This command allows you to run a specific test in a specific file. For example, suppose `file.py` has 100 test cases, and test case `test_case` fails. So, you would go and fix the issue. To check that you fixed this test, you could rerun all 100 tests, but it's faster to just rerun the one test case that was failing
```bash
$ pytest <options> file.py::test_case
```

Pytest has many useful options. One I use frequently is `-v`, which runs the test in verbose mode and allows you to see each test you are running more explicitly.

## Assert a mock object was called.
```python
mock = Mock()
mock.method(1, 2, 3, test='wow')
mock.assert_called() # true
```
If the method you are using is asynchronous, the you should use `assert_awaited`.
```python
mock = AsyncMock()
mock()
mock.assert_called() # true
mock.assert_awaited() # this will raise an error
```
Another example:
```python
mock = Mock(AsyncMock)
await mock.method()
mock.assert_called() # true
mock.assert_awaited() # true
```

## Assert a mock object was called exactly once 
Sometimes, you might want to ensure that something was called exactly once. Unittest 
has a thing for that, `assert_called_once`. If our code does this
```python
mock = Mock()
mock.method(1, 2, 3, test='wow')
mock.method(1, 2, 3, test='wow')
```
the next call will fail.
```python
mock.method.assert_called_once() 
```
The asynchronous version is `assert_awaited_once`.
```python
mock = AsyncMock()
await mock(1, 2, 3, test="wow")
mock.assert_awaited_once()
```


## Assert a mock object was called with known arguments
```python
mock = Mock()
mock.method(1, 2, 3, test='wow')
mock.method("foo", "bar")
```
This call will succeed.
```python
mock.method.assert_called_with(1, 2, 3, test='wow') 
```
The asynchronous version is `assert_awaited_with`.
This call will succeed.
```python
awaite mock.method(1, 2, 3, test='wow')
mock.method.assert_awaited_with(1, 2, 3, test='wow') 
```

## Get the arguments called by a mocked object 
```python
mock = Mock(return_value=None)
mock("A", "B", {1 : "meow"})
mock.call_args # => call('A', 'B', {1: 'meow'})
mock.call_args.args # => ('A', 'B', {1: 'meow'})
type(mock.call_args.args) # tuple
mock.call_args.args[0] # => "A"
mock.call_args.args[1] # => "B"
```

## Get the call args as a dictionary
The call args list can be casted to a dictionary if the function call 
is performed with keyword arguments 
For example, if we have a call like this
```python
params = {"foo": "bar"}
async with session.get(url, params=params) as response
    ...
```
Then we can mock the `session` object and its associated `session.get` method with the 
following code.
```python
mock_response = MockResponse(status=200)
session = mock.MagicMock(
    get=mock.MagicMock(
        return_value=mock.MagicMock(
            __aenter__=mock.AsyncMock(
                return_value=mock_response
            )
        )
    )
)
assert session.get.call_args.kwargs["params"]["foo"] == "bar"
```


## Assert called certain number of times
```python
mock = Mock()
mock.method(1, 2, 3, test='wow')
assert mock.method.call_count == 1
mock.method("foo")
assert mock.method.call_count == 2
mock.method("bar")
assert mock.method.call_count == 3
```

## Assert called once exactly
```python
mock = Mock()
mock.method(1, 2, 3, test='wow')
mock.method.assert_called_once()
```

## Assert not called called at all
```python
mock = Mock()
mock.method(1, 2, 3, test='wow') mock.some_other_method.assert_not_called()
```

## Parameterize a test with a list of arguments to run the test on
```python
@pytest.mark.parametrize(
    "arg,expected",
    [
        (5, 25),
        (6, 36),
        (7, 49),
    ],
)
async def my_test(
    arg,
    expected,
):
    assert raise_by_two(arg) == expected
```

## Mocking an aiohttp session call
Suppose you have a function making an aiohttp call like this.

```python
def contact_service(client_session):
    async with client_session.get(
        f"{SERVICE}/endpoint",
    ) as response:
        if response.status != 200:
            return None
        res = await response.json()

        item_a = res["foo"]
        item_b = res["bar"]

        return {
            "item_a": item_a, "item_b": item_b,
        }

    return None
```

The crux of writing a test for this is mocking the `client_session.get` call. Do it by using this 
Mock Response class and adding this to your tests
```python
class MockResponse:
    def __init__(self, json_data={}, status=200, text="", reason=""):
        self.json_data = json_data
        self.status = status
        self._text = text
        self.reason = reason

    async def json(self):
        return self.json_data

    async def text(self):
        return self._text

    def raise_for_status(self):
        pass

def mock_client_session_with_get(mock_response: MockResponse):
    return mock.MagicMock(
        get=mock.MagicMock(
            return_value=mock.MagicMock(
                __aenter__=mock.AsyncMock(
                    return_value=mock_response
                )
            )
        )
    )

@pytest.mark.asyncio
async def test_contact_service():
    mock_response = MockResponse(
        json_data={"item_a": "foo", "item_b": "bar"}, status=200
    )

    client_session=mock_client_session_with_get(mock_response)
    assert (
        await contact_service(client_session) is not None
    )

@pytest.mark.asyncio
async def test_contact_service_bad_response():
    # given 
    mock_response = MockResponse(status=400)
    client_session=mock_client_session_with_get(mock_response)
    assert await contact_service(client_session) is None
```

## Assert that an exception was raised when you called a function 
```python
def func():
    raise ValueError()
```
The test would then be
```python
import pytest

def test_func()
    with pytest.raises(ValuerError):
        func()
```

## Make a mock object raise an exception
To make a mock object raise an exception (to perhaps write a test for the function
which could receive an exception), use `side_effect`.
```python
async def operation_with_exception(foo: Object, bar: str)-> str:
    try:
        res = foo.get_something(bar)
    except Exception as e:
        return ""
    return "Cool!"
```
The test for `operation_with_exception` is then this:
```python

async def test_operation_with_exception_unhandled_exception():
    foo = mock.MagicMock(
        get_something=mock.MagicMock(side_effect=ValueError())
    )
    bar = "bar"

    # when
    res = await render_template(foo, bar)
    assert res == ""
```
Another example:
```python
mock = Mock(my_function=Mock(side_effect=KeyError('foo')))
mock.my_function() # this will raise an error!
```

## Call a mock object multiple times and have it return different values
This one is really handy. It is usually helpful for when you need to test some code that
calls a function twice in the same scope, and you're interested in testing all possible code 
paths (which would be: first call fails, or first call succeeds but second call fails, and both calls succeed). 

```python
mock = Mock()
mock.side_effect = [3, 2, 1]
assert mock() == 3
assert mock() == 2
assert mock() == 1
```


