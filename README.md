# DiffixValueGenerator

The ValueGenerator is a tool to generate synthetic data values from a database column using Diffix as offered by Aircloak. 

ValueGenerator queries the column through Diffix and stores certain column attributes. Subsequently, individual values may be requested that are syntactically similar to the original data. If the data is numeric, then values within a given range may be requested. If there are relatively few distinct values in the column, then the synthetic values are identical to those in the column. This tool does not preserve the statistical attributes of the original data.

## Prerequisites

* numpy
```
pip install numpy
```

* cloak username and password need to be set as environment variables

```
CLOAK_USER=username
CLOAK_PASS=password
```
* (optional, for experimental purposes) set raw database username and password.

```
RAW_USER=username
RAW_PASS=password
```

* Create the directory `training_data` under the working directory

## Usage

Code samples can be found in 
`examples.py`
