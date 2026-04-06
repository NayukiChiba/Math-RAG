"""API 占位路由可导入。"""


def test_register_routes_callable():
    from api.v1.router import registerRoutes

    assert registerRoutes() is None
