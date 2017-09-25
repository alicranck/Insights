'use strict'; // See note about 'use strict'; below

var myApp = angular.module('myAppLogin', [
  'ngMaterial', 'ui.router'
]);

myApp.controller('loginCtrl', ['$rootScope', '$scope', '$http', '$state', function($rootScope, $scope, $http, $state) {

  $scope.creds = {
    id: "",
    password: ""
  };

  $scope.message = '';

  $scope.loginUser = function() {
    $http({
      method: "POST",
      url: "/attempLogin",
      data: JSON.stringify($scope.creds),
      headers: {
        'Content-Type': 'application/json'
      }
    }).then(function(response) {
      if (response.data.status == 'OK') {
        $rootScope.user.id = $scope.creds.id;
        $rootScope.user.name = response.data.name;
        $rootScope.user.position = response.data.position;
        $rootScope.user.loggedIn = true;
        $state.go('getPatient');
      } else {
        $scope.message = response.data.message;
      }

    }, function(error) {
      $scope.message = "Something went wrong :(";
    })
  }


  /*  $scope.loginUser = function(creds) {
      AuthService.login(creds).then(function(user) {
        $rootScope.$broadcast(AUTH_EVENTS.loginSuccess);
        $scope.setCurrentUser(user);
      }, function() {
        $rootScope.$broadcast(AUTH_EVENTS.loginFailed);
      });
    };*/

}]);
