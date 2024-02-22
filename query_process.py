from documents import DocumentStore
from index import BaseIndex
from tokenizer import tokenize


def preprocess_query(query_str: str) -> list[str]:
    return tokenize(query_str)


def parse_phrases(query_terms: list[str]) -> dict[int, list[str]]:
    phrase_dict = dict()
    terms_in_phrase = []
    phrase_number = 0
    middle_of_quotes = False
    for term in query_terms:
        phrase_tracker = phrase_number
        in_phrase = False
        if term[0] == '"':
            term = term.lstrip('"')
            term = term.rstrip('"')
            terms_in_phrase.append(term)
            middle_of_quotes = True
            in_phrase = True
        elif term[len(term) - 1] == '"':
            term = term.rstrip('"')
            terms_in_phrase.append(term)
            middle_of_quotes = False
            phrase_number += 1
            terms_in_phrase = []
            in_phrase = True
        elif middle_of_quotes:
            terms_in_phrase.append(term)
            in_phrase = True

        if phrase_tracker == phrase_number and in_phrase:
            phrase_dict[phrase_number] = terms_in_phrase
    return phrase_dict


class FullDocumentOutputFormatter:
    def format_out(self, results: list[str], document_store: DocumentStore):
        output_string = ''
        for doc_id in results:
            doc = document_store.get_doc_by_id(doc_id)
            output_string += f'({doc.doc_id}) {doc.text}\n\n'
        return output_string


class DocIdsOnlyFormatter:
    def format_out(self, results: list[str], document_store: DocumentStore, unused_processed_query):
        return results


def format_out(results: list[str], document_store: DocumentStore, unused_processed_query) -> str:
    output_string = ''
    for doc_id in results:
        doc = document_store.get_doc_by_id(doc_id)
        output_string += f'({doc.doc_id}) {doc.text}\n\n'
    return output_string


class QueryProcess:
    def __init__(self, document_store: DocumentStore, index: BaseIndex, stopword_list_path: str, use_stopword: bool,
                 output_formatter=FullDocumentOutputFormatter()):
        self.document_store = document_store
        self.index = index
        self.stopword_list_path = stopword_list_path
        self.stopword_list = []
        self.use_stopword = use_stopword
        self.output_formatter = output_formatter

    def read_stopword_list(self):
        with open(self.stopword_list_path, 'r') as fp:
            for line in fp:
                line = line.strip()
                line = line.split("\"")
                if len(line) > 1:
                    self.stopword_list.append(line[1])

    def remove_stopwords(self, query: list[str]) -> list[str]:
        if len(self.stopword_list) == 0:
            self.read_stopword_list()
        new_query = []
        for term in query:
            if term not in self.stopword_list:
                new_query.append(term)
        return new_query

    def search(self, query: str, number_of_results: int) -> str:
        processed_query = preprocess_query(query)
        if self.use_stopword:
            processed_query = self.remove_stopwords(processed_query)
        print(processed_query)
        phrases = parse_phrases(processed_query)

        query_quotes_removed = []
        for term in processed_query:
            term = term.lstrip('"')
            term = term.rstrip('"')
            query_quotes_removed.append(term)
        processed_query = query_quotes_removed
        print(processed_query)

        results = self.index.search(processed_query, phrases, number_of_results)
        return self.output_formatter.format_out(results=results, document_store=self.document_store)
