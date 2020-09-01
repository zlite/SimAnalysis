from bokeh.server.server import Server
import thiel2

"""
create and run a demo bokeh app on a cloud server
"""





# configure and run bokeh server
kws = {'port': 5100, 'allow_websocket_origin': ['52.250.15.196'}
server = Server(thiel2, **kws)
server.start()
if __name__ == '__main__':
    server.io_loop.add_callback(server.show, '/')
    server.io_loop.start()