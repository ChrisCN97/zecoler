## Directory structure and naming convention

The data and metadata are organized in a rigorous directory structure. At the top level sits the `Project CodeNet` directory with several sub-directories, `data`, `metadata`, and `problem_descriptions`:

- `data` is further subdivided into a directory per problem and within each problem directory, directories for each language. The language directory contains all the source files supposed to be written in that particular programming or scripting language. When there are no submissions for a particular language, there will be no directory for it, but the problem directory will always be there, even if there are no submissions at all.

    The name of the directory for a programming language is the common name for the language using proper capitalization and special characters. This name is the consolidation of the names used in the metadata. Information is available about how the original language designations are mapped into the directory names and how these more general and common names are mapped to the submission file name extensions. As an example, a source could be designated c++14, which is mapped into the directory `C++` (notice the capital C) and will get the extension `.cpp`.
- `derived` holds information about near-duplicates, identical problem clusters, sample input and output for each problem, as well as the benchmarks. 
- `metadata` holds all the problem CSV files and the `problem_list.csv` file.
- `problem_descriptions` holds HTML files for most problems, giving an extensive description of the problem, often accompanied with some sample input and expected output.

For the sake of creating a uniform set of metadata across all data sources, and to hide any sensitive information, some metadata fields are anonymized by randomly (but uniquely and consistently) renumbering problem, submission, and user identifiers (ids). The identifiers we use are defined by simple regular expressions:

- problem ids are anonymized and follow this pattern: `p[0-9]{5}` (a `p` followed by exactly 5 digits).
- submission ids are anonymized and follow this pattern: `s[0-9]{9}` (an `s` followed by exactly 9 digits).
- user ids are anonymized and follow this pattern: `u[0-9]{9}` (a `u` followed by exactly 9 digits).

## Relationships among the metadata and data

The main relationship between problem metadata and data is the fact that each metadata record (a non-header row in a problem CSV file) describes one source file and provides all information about its location. The directory structure and naming convention as stated above are implicitly assumed.

### Example of getting the source file for a particular submission

Starting at a CSV metadata entry for a particular submission, here is how to get to the corresponding source file. Say that the submission id is `s300682070`. Either we know this is a submission to problem `p00001` upfront or we can grep through all `Project_CodeNet/metadata/p?????.csv` files to learn that. We get a brief description of this problem by looking at the `p00001` entry in the `Project_CodeNet/metadata/problem_list.csv`:

```console
p00001,List of Top 3 Hills,AIZU,1000,131072,,,
```

We can get a more verbose description of this problem by reading `Project_CodeNet/problem_descriptions/p00001.html`.

The `Project_CodeNet/metadata/p00001.csv` file provides the info on all submissions. For our selected submission we find:

```console
s300682070,p00001,u558442027,1480319506,JavaScript,JavaScript,js,Accepted,60,15496,219,4/4
```

We see it is an `Accepted` submission in the language `JavaScript` with file extension `.js`.

The source file path therefore is: `Project_CodeNet/data/p00001/JavaScript/s300682070.js`

### Example of getting the metadata for a particular source file

Likewise, we can play the reverse game of finding the metadata entry for a given submission source file. Say the source file is `Project_CodeNet/data/p00001/JavaScript/s300682070.js`.

Encoded in this file name path we see the problem id `p00001` and language `JavaScript` and of course the submission id `s300682070`. We find the metadata CSV file to be: `Project_CodeNet/metadata/p00001.csv`. Opening that file and searching for the submission id we find the entry:

```console
s300682070,p00001,u558442027,1480319506,JavaScript,JavaScript,js,Accepted,60,15496,219,4/4
```

### Dataset statistics

The dataset comprises 13,916,868 submissions, divided into 4053 problems (of which 5 are empty). Of the submissions 53.6% (7,460,588) are *accepted*, 29.5% are marked as *wrong answer* and the remaining suffer from one of the possible rejection causes. The data contains submissions in 55 different languages, although 95% of them are coded in the six most common languages (C++, Python, Java, C, Ruby, C#). C++ is the most common language with 8,008,527 submissions (57% of the total) of which 4,353,049 are *accepted*. 

## Data

The data consist of complete programs in a particular programming language. Each program is contained in a single file. The file will have a name with an extension that denotes the programming language used. (More details about the specific programming language and the version of the compiler/interpreter used, can be found in the metadata.)

Each program is a solution to a certain programming task or problem. There will be many problems and each problem might have many solutions in different languages. Solutions do not necessarily have to be complete and correct. Solutions can also be attempts at solving a problem, but for various reasons as indicated by the metadata fail to do so. Therefore we prefer to talk about submissions and not solutions in the sequel. Solutions are the accepted submissions which consist of compilable and executable programs that at least correctly produce the expected results on all provided test cases. (Of course, according to the late Dijkstra, tests are no proof of correctness.)

## Metadata

The metadata provides properties of interest about the problems and their submissions. Foremost it formalizes the organization of the data and the relationship between problems, languages, and the source code files. The metadata allows for queries about the data and to make specific selections among the large collection of problems, languages, and source files.

Metadata is made available in comma-separated value (CSV) files. This allows for easy processing, even with simple command-line tools. Some of the fields in the csv files might be empty, and for submissions that are not accepted, some fields might have invalid entries such as negative numbers for CPU time. Extra checking needs to be implemented in parsing these files.

The metadata is hierarchically organized on 2 levels: the first level is the dataset level that relates to all the different problems defined by the various dataset sources. The second level is the problem level that relates to all source code submissions pertaining to a single problem or task.

Metadata and data are deliberately kept fully separated within the file system.

### Metadata at the dataset level

At the dataset level there is a single CSV file (`problem_list.csv`) listing all the different problems. Additionally, for each problem there is a more extensive description that sets the problem and any further requirements and constraints and often provides examples of data input and expected output.

The fields and their format of this CSV file are captured by the following table:

name of column | data type | unit | description
-- | -- | -- | --
id           | string | none | unique anonymized id of the problem
name         | string | none | short name of the problem
dataset      | string | none | original dataset, AIZU or AtCoder
time_limit   | int    | millisecond | maximum time allowed for a submission
memory_limit | int    | KB | maximum memory allowed for a submission
rating       | int    | none | rating, i.e., difficulty, of the problem
tags         | string | none | list of tags separated by "\|"; not used
complexity   | string | none | degree of difficulty of the problem; not used

### Metadata at the problem level

At the problem level there is a CSV file per problem and all content of these files is of course organized under one and the same header.

The fields and their format of this CSV file are captured by the following table:

name of column | data type | unit | description
-- | -- | -- | --
submission_id | string | none | unique anonymized id of the submission
problem_id    | string | none | anonymized id of the problem
user_id       | string | none | anonymized user id of the submission
date          | int    | seconds | date and time of submission in the Unix timestamp format (seconds since the epoch)
language      | string | none | mapped language of the submission (ex: C++14 -> C++)
original_language | string | none | original language specification
filename_ext  | string | none | extension of the filename that indicates the programming language used
status        | string | none | acceptance status, or error type
cpu_time      | int    | millisecond | execution time
memory        | int    | KB | memory used
code_size     | int    | Bytes | size of the submission source code in bytes
accuracy      | string | none | number of tests passed; *Only for AIZU

Here is a table of all the possible status values. The “abbreviation” and “numeric code” are sometimes seen in the original metadata on the websites; it is listed here for reference and completeness. These fields do not occur in the Project CodeNet metadata.

status | abbreviation | numeric code
-- | -- | --
Compile Error          | CE  |  0
Wrong Answer           | WA  |  1
Time Limit Exceeded    | TLE |  2
Memory Limit Exceeded  | MLE |  3
Accepted               | AC  |  4
Judge Not Available    | JNA |  5
Output Limit Exceeded  | OLE |  6
Runtime Error          | RE  |  7
WA: Presentation Error | PE  |  8
Waiting for Judging    | WJ  | 
Waiting for Re-judging | WR  | 
Internal Error         | IE  | 
Judge System Error     |     | 


