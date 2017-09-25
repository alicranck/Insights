.factory('AuthService', function($http, Session) {
    var authService = {};

    authService.login = function(credentials) {
      var jsonCreds = JSON.stringify(credentials);
      $http({
        method: "POST",
        url: "/loginUser",
        data: jsonCreds,
        headers: {
          'Content-Type': 'application/json'
        }
      }).then(function(response) {
        $scope.status = response.status;
      }, function(error) {
        $scope.message = error;
      })
    });

  authService.isAuth = function(user) {
    return user.loggedIn;
  });
};
