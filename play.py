#!/usr/bin/env python3
import time
import traceback
import chess
import chess.svg
import torch
from static.state import State
from training.train import Net
from flask import Flask, Response, request, render_template


class Valuator(object):
    def __init__(self):
        vals = torch.load("nets/value.pth", map_location=lambda storage, loc: storage)
        self.model = Net()
        self.model.load_state_dict(vals)

    def __call__(self, s):
        brd = s.serialize()[None]
        output = self.model(torch.tensor(brd).float())
        return float(output.data[0][0])
def explore_leaves(s, v):
    ret = []
    for e in s.edges():
        s.board.push(e)
        ret.append((v(s), e))
        s.board.pop()
    return ret


# chess board and "engine"
s = State()
v = Valuator()

app = Flask(__name__)


@app.route('/')
def hello():
    html_content = '<html><head>'
    html_content += '<meta charset="UTF-8"> <title>Chess with Neural network</title>'
    html_content += '<center> <h1> Project FASI :</h1>'
    html_content += '<style>input { font-size: 30px; } button { font-size: 30px; }' \
                    '       image{ padding:50 px;}' \
                    'body {  background-color:#CCD8D0;}</style>'
    html_content += '</head><body>'
    html_content += '<img width=480 height=480 src="/board.svg?%f"></img><br/>' % time.time()
    html_content += '<form action="/move"><input name="move"  type="text"> </input><input type="submit" value="Your ' \
                    'Move "></form><br/> </center>'
    html_content += '<footer>\
    <div class="container">\
        <div class="row">\
            <div class="col-sm-5 copyright">\
                <center><p>\
                    <h6>Copyright :</h6>  Mahamdi Mohammed &&  Mohamed-Hicham LEGHETTAS \
                </p></center>\
            </div>\
        </div>\
    </div>\
</footer>'
    return html_content


@app.route("/board.svg")
def board():
    return Response(chess.svg.board(board=s.board), mimetype='image/svg+xml')


def computer_move():
    # computer move
    move = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)[0]
    print(move)
    s.board.push(move[1])


@app.route("/move")
def move():
    if not s.board.is_game_over():
        move = request.args.get('move', default="")
        if move is not None and move != "":
            # print("human moves", move)
            try:
                s.board.push_san(move)
                computer_move()
            except Exception:
                traceback.print_exc()
    else:
        print("GAME IS OVER")
    return hello()


if __name__ == "__main__":
    app.run(debug=True)
