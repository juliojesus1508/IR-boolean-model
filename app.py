from flask import Flask
from flask import render_template, request

from context import *

app = Flask(__name__)

files = filename

model = BooleanModel()
BooleanModel.set_db_documents(model)
BooleanModel.import_container(model)


@app.route('/search_query')
def query_input(result=None):
    return render_template('request.html')

@app.route('/results', methods=['GET'])
def query_process(result=None):
    if request.args.get('query', None):                
        query = request.args['query']
        ir = BooleanModel.information_retrieval(model, query)        
        return render_template('response.html', result=ir)

if __name__ == '__main__':
    app.run(port= 3000, debug=True)
