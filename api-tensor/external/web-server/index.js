var http = require('http')

http.createServer(function (request, response) {
    // Send the HTTP header 
    // HTTP Status: 200 : OK
    // Content Type: text/plain


    var allowedLogins = [
        {
            "Username": "@MontrayDavis",
            "Password": "abc123"
        },
        {
            "Username": "@DemoUser1",
            "Password": "abc456"
        },
        {
            "Username": "@DemoUser2",
            "Password": "abc789"
        }
    ]

    var data = "";
    request.on('data', chunk => {
        data += chunk;
    });

    request.on('end', function () {
        try {
            var loginInfo = JSON.parse(data);

            var matches = allowedLogins
                .filter(a => a['Username'] == loginInfo['Username'] && a['Password'] == loginInfo['Password']);

            var responseObject = {
                "Success": (matches.length > 0)
            };

            response.end(JSON.stringify(responseObject));
        } catch (err) {
            response.end(err);
        }
    })

    response.writeHead(200, { 'Content-Type': 'text/plain' });

    // Send the response body as "Hello World"

}).listen(8081);

// Console will print the message
console.log('Server running at http://127.0.0.1:8081/');