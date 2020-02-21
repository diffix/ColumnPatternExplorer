import json
import os
import pprint
import sys
import time
from decimal import Decimal

import numpy as np
import psycopg2

import utils


class openValGenerator:

    def __init__(self, db, table, column, host='attack.aircloak.com', port=9432, cloak_only=True):
        self._pp = pprint.PrettyPrinter(indent=2)
        self.host = host
        self.port = port
        self.db = db
        self.table = table
        self.column = column
        self.__length_frequency = {}
        self.__chars_count = {}
        self.__strings_count = {}
        self.trained = False
        self.__column_path = f'{os.path.dirname(os.path.abspath(sys.argv[0]))}/training_data/{self.db}.{self.table}.{self.column}.json '
        self.__email = False
        self.cloak_user, self.cloak_password, self.raw_user, self.raw_password = utils.getEnvVars(
            getRaw=not cloak_only)
        self.column_type = self.get_column_type()
        self.cloak_only = cloak_only

    def istrained(self):
        self.trained = os.path.isfile(self.__column_path)
        if self.trained:
            with open(self.__column_path, "r") as read_file:
                self.__chars_count = json.load(read_file)
        same_db = 'cloak_only' in self.__chars_count and self.__chars_count[
            'cloak_only'] == self.cloak_only
        return self.trained and same_db

    def getChars(self):
        anonConnStr = f'''
            host={self.host} 
            port={self.port} 
            dbname={self.db} 
            user={self.cloak_user} 
            password={self.cloak_password}
            '''
        anonConn = psycopg2.connect(anonConnStr)
        anonCur = anonConn.cursor()
        if not self.cloak_only:
            rawConnStr = f'''
                host={self.host} 
                port={self.port} 
                dbname={self.db} 
                user={self.raw_user} 
                password={self.raw_password}'''
            rawConn = psycopg2.connect(rawConnStr)
            rawCur = rawConn.cursor()
        char_count = {"chars_1": [], "bias_1": [],
                      "chars_2": [], "bias_2": [],
                      "chars_left": [], "bias_left": [],
                      "chars_right": [], "bias_right": [],
                      "strings": [], "counts": [],
                      "ats": 0, "dots": 0, "total_count": 0,
                      "hidden": 0, "has_single_at": 0, 'cloak_only': self.cloak_only}

        # query the DB for all the strings in the column
        anonCur.execute(
            f"""select {self.column}, count(*) 
                        FROM 
                        {self.table}
                        GROUP 
                        BY 1;""")
        data = anonCur.fetchall()
        if data[0][0] == '*':
            data.pop(0)
        strings = []
        count = []
        for row in data:
            strings.append(row[0])
            count.append((row[1]))
        # Query the DB for the total number of rows
        anonCur.execute(f"select count(*) FROM {self.table}")
        data = anonCur.fetchone()
        char_count["total_count"] = data[0]
        char_count["hidden"] = char_count["total_count"] - sum(count)
        char_count["strings"] = strings
        char_count["counts"] = count
        ###############################################################################################################
        # train on substrings of length = 1 (single characters)
        i = 1
        while True:
            anonCur.execute(f'''
                SELECT substring({self.column},{i},1), count(*) 
                FROM {self.table} 
                GROUP BY 1''')
            data = anonCur.fetchall()
            # column doesn't have any string with length i : terminate
            if data[0][0] == '' and len(data) == 1:
                break
            if data[0][0] == '*':
                data.pop(0)
            b = []
            c = []
            for row in data:
                char_count["ats"] = char_count["ats"] + \
                    row[1] if row[0] == '@' else char_count["ats"]
                char_count["dots"] = char_count["dots"] + \
                    row[1] if row[0] == '.' else char_count["dots"]
                c.append(row[0])
                b.append(float(row[1]))
            b = np.array(b, dtype=np.float64)
            b /= b.sum()
            char_count["chars_1"].append(c)
            char_count["bias_1"].append(b.tolist())
            i += 1

        # train on substrings of length = 2
        ###############################################################################################################
        i = 1
        while True:
            anonCur.execute(f"""
                SELECT substring({self.column},{i},2), count(*) 
                FROM {self.table} 
                GROUP BY 1""")
            data = anonCur.fetchall()
            # column doesn't have any string with length = i => terminate
            if data[0][0] == '' and len(data) == 1:
                break
            if data[0][0] == '*':
                data.pop(0)
            b = []
            c = []
            for row in data:
                c.append(row[0])
                b.append(float(row[1]))
            b = np.array(b, dtype=np.float64)
            b /= b.sum()
            char_count["chars_2"].append(c)
            char_count["bias_2"].append(b.tolist())
            i += 1

        # dynamic substring discovery (left)
        ###############################################################################################################
        i = 1
        j = 1
        last_b = []
        last_c = []
        while True:
            anonCur.execute(f"""
                SELECT ss, sum(cnt)
                FROM (
                SELECT substring({self.column},{i},{j}) as ss, count(*) as cnt
                FROM {self.table} 
                GROUP BY 1) t
                GROUP BY 1;
                """)
            data = anonCur.fetchall()
            # column doesn't have any string with length = i => terminate
            if str(data[0][0]).strip() == '' and len(data) == 1:
                return char_count
            if data[0][0] == '*':
                data.pop(0)
            c = []
            b = []
            for row in data:
                c.append(row[0])
                b.append(row[1])
            if np.array_equal(last_c, c) and np.array_equal(last_b, b):
                b = np.array(b, dtype=np.float64)
                b /= b.sum()
                char_count["chars_left"].append(c)
                char_count["bias_left"].append(b.tolist())
                i = max(i + 1, i + j - 1)
                j = 1
                break
            frequent = False
            b = np.array(b, dtype=np.float64)
            sb = sum(b)
            for x, y in zip(c, b):
                # y is a common substring and thers is no hidden substrings
                if y >= (sb * 0.1) and x != '' and sb == char_count["total_count"]:
                    j += 1
                    last_c = c
                    last_b = b
                    frequent = True
                    break
            if frequent:
                continue
            b = np.array(b, dtype=np.float64)
            b /= b.sum()
            char_count["chars_left"].append(c)
            char_count["bias_left"].append(b.tolist())
            i = max(i + 1, i + j - 1)
            j = 1

        if not self.cloak_only:
            rawCur.execute(
                f"SELECT {self.column} FROM {self.table} WHERE {self.column} like '%@%@%'")
            data = rawCur.fetchall()
            char_count["has_single_at"] = char_count["total_count"] - len(data)
            ###########################################################################################################
            # dynamic substring discovery (right) cannot be done through the cloak
            i = 1
            j = 1
            last_b = []
            last_c = []
            while True:
                rawCur.execute(f"""
                                   SELECT ss, sum(cnt)
                                   FROM (
                                   SELECT substring(reverse({self.column}),{i},{j}) as ss, count(*) as cnt
                                   FROM {self.table} 
                                   GROUP BY 1
                                   HAVING count(distinct uid) > 3) t
                                   GROUP BY 1;
                                   """)
                data = rawCur.fetchall()
                # column doesn't have any string with length = i => terminate
                if str(data[0][0]).strip() == '' and len(data) == 1:
                    break
                if data[0][0] == '*':
                    data.pop(0)
                c = []
                b = []
                for row in data:
                    c.append(''.join(reversed(row[0])))
                    b.append(row[1])
                if np.array_equal(last_c, c) and np.array_equal(last_b, b):
                    b = np.array(b, dtype=np.float64)
                    b /= b.sum()
                    char_count["chars_right"].append(c)
                    char_count["bias_right"].append(b.tolist())
                    i = max(i + 1, i + j - 1)
                    j = 1
                    break
                frequent = False
                b = np.array(b, dtype=np.float64)
                sb = sum(b)
                for x, y in zip(c, b):
                    # y is a common substring and there are no hidden substrings
                    if y >= (sb * 0.1) and x != '' and sb == char_count["total_count"]:
                        j += 1
                        last_c = c.copy()
                        last_b = b.copy()
                        frequent = True
                        break
                if frequent:
                    continue
                b = np.array(b, dtype=np.float64)
                b /= b.sum()
                char_count["chars_right"].append(c)
                char_count["bias_right"].append(b.tolist())
                i = max(i + 1, i + j - 1)
                j = 1
            rawConn.close()
        return char_count

    def is_email(self):
        if not self.__chars_count:
            with open(self.__column_path, "r") as read_file:
                self.__chars_count = json.load(read_file)
        ats = self.__chars_count["ats"]
        dots = self.__chars_count["dots"]
        total_count = self.__chars_count["total_count"]
        has_single_at = (
            self.__chars_count['has_single_at']/total_count >= 0.99) or self.cloak_only
        self.__email = (ats / total_count >= 0.99) and (dots /
                                                        total_count >= 1) and has_single_at
        return self.__email

    def get_column_type(self):
        connStr = f'''
            host={self.host}
            port={self.port}
            dbname={self.db} 
            user={self.cloak_user}
            password={self.cloak_password}'''
        conn = psycopg2.connect(connStr)
        cur = conn.cursor()
        query = f"SHOW COLUMNS FROM {self.table};"
        cur.execute(query)
        data = cur.fetchall()
        conn.close()
        for row in data:
            if row[0] == self.column:
                self.column_type = row[1]
                return self.column_type

    def train_numeric(self):
        start = time.time()
        print(f"Started Training {self.column}...")
        connStr = f'''
                    host={self.host}
                    port={self.port} 
                    dbname={self.db} 
                    user={self.cloak_user} 
                    password={self.cloak_password}
                    '''
        conn = psycopg2.connect(connStr)
        cur = conn.cursor()
        cur.execute(
            f"""select {self.column}, count(*)
                    FROM {self.table}
                    GROUP BY 1;
                    """)
        data = cur.fetchall()
        if len(data) > 0 and data[0][0] is None:
            data.pop(0)
        __length_frequency = {}
        # Default procedure
        for row in data:
            has_decimal_point = len(str(row[0]).split('.')) > 1
            decimal_part = str(Decimal(str(row[0])) % 1)
            decimal_part_length = len(decimal_part) - \
                2 if has_decimal_point else 0
            # decimal_part_length = decimal_part_length if Decimal(decimal_part) > 0 else 1
            # print(row,decimal_part,decimal_part_length,has_decimal_point)
            if decimal_part_length in __length_frequency:
                __length_frequency[decimal_part_length] += row[1]
            else:
                __length_frequency[decimal_part_length] = row[1]
        table_lengths = list(__length_frequency.keys())
        counts = list(__length_frequency.values())
        table_counts = np.array(counts, dtype=np.float64)
        table_counts /= table_counts.sum()
        positions = {}
        dots = 0
        i = 1
        # Not sure what the following is for, but anyway it is buggy, so
        # leave it for now
        cur.execute(f"select count(*) FROM {self.table}")
        data = cur.fetchone()
        total_count = data[0]
        total_derived = sum(__length_frequency.values())
        if (total_derived / total_count) < 0.75:
            while True:
                # print(f"{__length_frequency}")
                sql = f"""
                            SELECT substring(CAST ({self.column} AS TEXT),{i},1), count(*) 
                            FROM {self.table} 
                            GROUP BY 1"""
                # print(sql)
                cur.execute(sql)
                data = cur.fetchall()
                # self._pp.pprint(data)
                # column doesn't have any string with length i : terminate
                if data[0][0] == '' and len(data) == 1:
                    break
                if data[0][0] == '*':
                    data.pop(0)
                for row in data:
                    if row[0] == ".":
                        dots = dots + row[1]
                        positions[i] = row[1]
                i += 1
            max_length = i - 1
            for position in positions.keys():
                # print(f"max_length {max_length}, position {position}, diff {max_length-position}")
                if (max_length - position) in __length_frequency:
                    __length_frequency[max_length -
                                       position] += positions[position]
                else:
                    __length_frequency[max_length -
                                       position] = positions[position]
            integers = max(total_count - dots, 0)
            print(total_count, dots, integers)
            if 0 in __length_frequency:
                __length_frequency[0] += integers
            else:
                __length_frequency[0] = integers
        conn.close()
        __length_frequency['cloak_only'] = self.cloak_only
        with open(self.__column_path, 'w') as outfile:
            json.dump(__length_frequency, outfile, indent=2)
        end = time.time()
        print(f"Completed Training in: {round(end - start)} s")

    def train(self):
        if self.column_type == "text":
            self.train_text()
        if self.column_type in ["real", 'integer']:
            self.train_numeric()

    def getVal(self, low=None, high=None, mode='left'):
        if self.column_type == "text":
            return self.getVal_text(mode)
        if self.column_type in ["real", 'integer']:
            return self.getVal_numeric(low, high)

    def getVal_numeric(self, low, high):
        if not self.__length_frequency:
            with open(self.__column_path, "r") as read_file:
                self.__length_frequency = json.load(read_file)
        self.__length_frequency.pop('cloak_only', None)
        lengths = list(self.__length_frequency.keys())
        counts = list(self.__length_frequency.values())
        lengths = np.array(lengths, dtype=np.int64)
        counts = np.array(counts, dtype=np.float64)
        counts /= counts.sum()
        values = np.linspace(low, high)
        generated = []
        for x in range(len(values)):
            length = utils.choice(lengths, p=counts)
            value = np.round(utils.getRandomVal(values), length) if length > 0 else int(
                utils.getRandomVal(values))
            generated.append(value)
        return utils.getRandomVal(generated)

    def generateString(self, mode):
        if mode == 1:
            string = ""
            for i in range(len(self.__chars_count[f'chars_1'])):
                char = utils.getCharacter(self.__chars_count[f'chars_1'][i], np.array(
                    self.__chars_count[f'bias_1'][i]))
                if char == '':
                    return string
                string += char
            return string
        elif mode == 2:
            chars = self.__chars_count[f"chars_{2}"]
            bias = self.__chars_count[f"bias_{2}"]
            bias[0] = np.array(bias[0])
            bias[0] /= bias[0].sum()
            string = utils.getCharacter(chars[0], bias[0])
            n = len(chars)
            i = 1
            while i < n - 1:
                bias[i] = np.array(bias[i])
                filtered_chars, filtered_bias = [], []
                for c, b in zip(chars[i], bias[i]):
                    if len(c) > 0 and c[0] == string[-1]:
                        filtered_chars.append(c)
                        filtered_bias.append(b)
                if filtered_chars:
                    filtered_bias = np.array(filtered_bias)
                    filtered_bias /= filtered_bias.sum()
                    c = utils.getCharacter(filtered_chars, filtered_bias)
                    if len(c) == 2:
                        string += c[1]
                    else:
                        return string
                else:
                    return string
                i += 1
            return string
        elif mode == "right":
            string = ""
            for i in range(len(self.__chars_count[f'chars_right'])):
                char = utils.getCharacter(self.__chars_count[f'chars_right'][i],
                                          np.array(self.__chars_count[f'bias_right'][i]))
                if char == '':
                    return string
                string = char + string
        elif mode == 'left':
            string = ""
            for i in range(len(self.__chars_count[f'chars_left'])):
                char = utils.getCharacter(self.__chars_count[f'chars_left'][i],
                                          np.array(self.__chars_count[f'bias_left'][i]))
                if char == '':
                    return string
                string += char
            return string

    def getVal_text(self, mode):
        if mode == 'right' and self.cloak_only:
            print('Right mode can not be used in cloak only mode')
            return
        if not self.__chars_count:
            with open(self.__column_path, "r") as read_file:
                self.__chars_count = json.load(read_file)
                self.__email = self.is_email()
        options = ["synthetic", "fromCloak"]
        prob = [self.__chars_count["hidden"] / self.__chars_count["total_count"],
                1 - (self.__chars_count["hidden"] / self.__chars_count["total_count"])]
        option = utils.choice(options, p=prob)
        if option == "synthetic":
            generated_str = self.generateString(mode)
            while self.__email and not utils.is_valid_address(generated_str):
                generated_str = self.generateString(mode)
            if self.__email:
                generated_str = generated_str.replace('.....', '.')
                generated_str = generated_str.replace('....', '.')
                generated_str = generated_str.replace('...', '.')
                generated_str = generated_str.replace('..', '.')
                generated_str = generated_str.replace('.@', '@')
                generated_str = generated_str.replace('@.', '@')
                if generated_str[-1] == ".":
                    generated_str = generated_str[:-1]
        else:
            generated_str = utils.getRandomString(
                self.__chars_count["strings"], self.__chars_count["counts"])
        return generated_str

    def train_text(self):
        print("Started Training...")
        start = time.time()
        self.__chars_count = self.getChars()
        with open(self.__column_path, 'w') as outfile:
            json.dump(self.__chars_count, outfile, indent=2)
        self.__email = self.is_email()
        end = time.time()
        print(f"Completed Training in: {round(end - start)} s")
