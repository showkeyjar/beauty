<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<title></title>
	<link rel="stylesheet" type="text/css" href="main.css">
</head>
<body>

	<p><h3>留言内容</h3></p>
	<div class="comment">
		<?php
		// 展示留言数据 并作分页显示

		include_once 'conn.php';
		include_once 'config.php';

		$p = isset($_GET['p'])?$_GET['p']:1;

		$offset = ($p-1)*$pageSize;

		$sql_query = "select * from guestbook order by id desc limit $offset,$pageSize";
		$query_result = mysql_query($sql_query);
		if (!$query_result) {//对于select语句的查询 如果成功返回一个资源标识符 如果失败返回false
			exit("查询数据错误：".mysql_error());
		}


		while ($gb_array = mysql_fetch_array($query_result)) {
			$nickname = $gb_array['nickname'];
			$content = nl2br($gb_array['content']); //nl2br:在字符串中每个新行(\n)之前都插入HTML换行符<br/>
			$createtime = date('Y-m-d H:i',$gb_array['createtime']);
			echo $nickname." 发表于 ".$createtime."</br></br>";
			echo "内容：".$content."<br/><br/>";

			if ($gb_array['reply']) {

				?>

				<div class="replyDiv">
					管理员回复时间：<?php echo date("Y-h-d H:i",$gb_array['replytime']);?><br/>
					回复内容为：<?=nl2br($gb_array['reply']);?><br/><br/>
				</div>

			<?php
			}
			echo "<hr/>";
		}

		$count_result = mysql_query("select count(*) from guestbook");
		$count_array = mysql_fetch_array($count_result);
		$pageNum = ceil($count_array['count(*)']/$pageSize);

		echo "共".$count_array['count(*)']."条留言：";	

		if ($pageNum>1) {
			for ($i=1; $i <= $pageNum; $i++) { 
				if ($i == $p) {
					echo '['.$i.']';
				}else{
					echo "&nbsp;<a href='index.php?p=$i'>$i</a>&nbsp;";
				}
			}
		}
		?>
	</div>
	
	<div class="FormDiv">
		<h3>发表留言</h3>
		<form action="submitting.php" method="post" name="LeaveMessage" onsubmit="return InputCheck(this);return InputComment(this);">
			<p>
				<label for="nickname" class="label">昵&nbsp;&nbsp;称：</label>
				<input type="text" name="nickname" id="nickname" class="input">
				<span>(必须填写，不超过16个字符串)</span>
			</p>

			<p>
				<label for="email" class="label">电子邮箱：</label>
				<input type="text" name='email' id="email" class="input">
				<span>(非必须，不超过60个字符串)</span>
			</p>

			<p>
				<label for="content" class="label">留言内容：</label>
				<textarea name="content" id="content" cols="50" rows="10"></textarea>
			</p>
			
			<p>
				<input type="submit" name="submit" class="submit" value="确定"></input>
			</p>

		</form>
	</div>
</body>
<script language="javascript" src="http://localhost/js/jquery.js"></script>
<script type="text/javascript">
	function InputCheck(LeaveMessage){
		if (LeaveMessage.nickname.value == '') {
			alert('昵称不能为空');
			LeaveMessage.nickname.focus();
			return false;
		}

		if (LeaveMessage.content.value == '') {
			alert('留言内容不能为空');
			LeaveMessage.content.focus();
			return false;//函数返回值为false 则不会提交到submitting.php后台处理界面
		}

	}

</script>
</html>





