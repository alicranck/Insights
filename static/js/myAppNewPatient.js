'use strict';   // See note about 'use strict'; below

var myApp = angular.module('myAppNewPatient', [
'ngMaterial'
]);

myApp.controller('newCtrl',['$scope','$http',function ($scope,$http){

  $scope.patient = {
    id:"",
    firstName:"" ,
    lastName:""
  } ;

  $scope.message='';


  $scope.insertPatient = function(patient){
    var jsonPatient = JSON.stringify($scope.patient) ;
    $http({
      method: "POST" ,
      url: "/insertPatient" ,
      data:jsonPatient,
      headers:{'Content-Type': 'application/json'}
    }).then(function(response){
    $scope.message = response ;
    }
      ,function(error){
        $scope.message = error ;
      }
    )
  };

}]);
