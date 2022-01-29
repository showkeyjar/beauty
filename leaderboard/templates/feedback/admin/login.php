</html>

<?php
session_start();
include '../conn.php';

if ($_POST) {
	$username = htmlspecialchars($_POST['username']);
	$password = MD5($_POST['password']);

	$check_sql = "select * from user where username='$username' and password='$password'";
	$query_result = mysql_query($check_sql);
	if ($query_array = mysql_fetch_array($query_result)) {
		$_SESSION['username'] = $query_array['username'];
		$_SESSION['password'] = $query_array['password'];
		$_SESSION['id'] = $query_array['id'];
		
		header("Location:http://localhost/project/myguestbook/admin/admin.php");
	}else{
		echo "密码错误".mysql_error();
	}
}

?>
<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<title>登录</title>
	<style type="text/css">
		fieldset{
			width: 420px;
			margin: 0 auto;
			margin-top: 100px;
			font: 13px normal Times;
		}

		legend{
			color: blue;
		}
		.label{
			float: left;
			margin-left: 20px;
			width: 80px;
		}

		.input{
			width: 50%;
		}

		.submit{
			float: left;
			margin-top:20px;
			margin-left: 100px;
			width: 80px;
			height: 30px;
			color: maroon;
			font-size: 14px;
			letter-spacing: 10px;
			text-align: center;
			text-indent: 11px;
			cursor: pointer;
		}
	</style>
</head>
<body>
		<fieldset>
			<legend><h3>用户登录</h3></legend>
			<form action="login.php" method="post" name="LoginForm" onsubmit="return InputCheck(this);">
				<p>
					<br/>
					<label class="label" for="username">用户名：</label>
					<input type="text" id="username" name="username" value="admin" class="input">
				</p>

				<p>
					<label for="psssword" class="label">密码：</label>
					<input type="password" id="input" name="password" class="input">
				</p>

				<p>
					<input type="submit" name="submit" class="submit" value="确定">
				</p>
			</form>
		</fieldset>
</body>
<script type="text/javascript">
	function InputCheck(LoginForm) {
		if (LoginForm.username.value == '') {
			alert('用户名不能为空');
			LoginForm.username.focus();
			return false;
		}

		if (LoginForm.password.value == '') {
			alert('密码不能为空!');
			LoginForm.password.focus();
			return false;
		}
	}
</script>
